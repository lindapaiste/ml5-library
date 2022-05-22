// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* eslint max-len: "off" */
/*
Pix2pix
The original pix2pix TensorFlow implementation was made by affinelayer: github.com/affinelayer/pix2pix-tensorflow
This version is heavily based on Christopher Hesse TensorFlow.js implementation: https://github.com/affinelayer/pix2pix-tensorflow/tree/master/server
*/

import * as tf from '@tensorflow/tfjs';
import callCallback from '../utils/callcallback';
import CheckpointLoaderPix2pix from '../utils/checkpointLoaderPix2pix';
import { array3DToImage } from '../utils/imageUtilities';

// TODO: can this be written as a layersModel?
class PictModel {
  constructor(variables) {
    /**
     * @type {Record<string, tf.Tensor>}
     */
    this.variables = variables;
    /**
     * @type {tf.Tensor[]}
     */
    this.layers = [];
    /**
     * @type {number}
     */
    this.currentStep = 1;
    /**
     * @type {boolean}
     */
    this.isEncoding = true;
  }

  reset() {
    this.layers = [];
    this.currentStep = 1;
    this.isEncoding = true;
  }

  getVariable(path) {
    const scope = `${this.isEncoding ? 'encoder' : 'decoder'}_${this.currentStep}`;
    const value = this.variables[`generator/${scope}/${path}`];
    if (!value) {
      console.error(`generator/${scope}/${path}`, Object.keys(this.variables));
    }
    return value;
  }

  maybeBatchNorm(input) {
    if (this.currentStep <= 1) {
      return input;
    }
    const scale = this.getVariable('batch_normalization/gamma');
    const offset = this.getVariable('batch_normalization/beta');
    return tf.tidy(() => {
      // TODO: why are variables moving_mean and moving_variance not used?
      const { mean, variance } = tf.moments(input, [0, 1]);
      const varianceEpsilon = 1e-5;
      return tf.batchNorm(input, mean, variance, offset, scale, varianceEpsilon);
    });
  }

  encode() {
    const filter = this.getVariable('conv2d/kernel');
    // TODO: why isn't the bias used?
    // const bias = this.getVariable('conv2d/bias');
    const layerInput = this.layers[this.currentStep - 1];
    return tf.tidy(() => {
      // Apply leakyRelu, but not on the input.
      const rectified = this.currentStep === 1 ? layerInput : tf.leakyRelu(layerInput, 0.2);
      // Apply conv2d.
      const convolved = tf.conv2d(rectified, filter, [2, 2], 'same');
      // Apply batchNorm, but not on the first layer.
      return this.maybeBatchNorm(convolved);
    });
  }

  decode() {
    return tf.tidy(() => {
      const lastLayer = this.layers[this.layers.length - 1];
      const skipLayer = this.layers[this.currentStep];
      console.log(this.currentStep, this.layers.length);
      const layerInput = this.currentStep === 8
        ? lastLayer
        : tf.concat([lastLayer, skipLayer], 2);
      const rectified = tf.relu(layerInput);
      const filter = this.getVariable('conv2d_transpose/kernel');
      const bias = this.getVariable('conv2d_transpose/bias');
      console.log('input', rectified.shape, 'filter', filter.shape)
      const convolved = tf.conv2dTranspose(rectified, filter, [rectified.shape[0] * 2, rectified.shape[1] * 2, filter.shape[2]], [2, 2], 'same');
      const biased = tf.add(convolved, bias);
      return this.maybeBatchNorm(biased);
    });
  }

  predict(preprocessedInput) {
    this.reset();
    this.layers[0] = preprocessedInput;
    // Note: this assumes that there are always 8 layers. is that always true?
    for (let i = 1; i <= 8; i += 1) {
      this.currentStep = i;
      this.layers[i] = this.encode();
    }
    this.isEncoding = false;
    for (let i = 8; i >= 1; i -= 1) {
      this.currentStep = i;
      this.layers.push(this.decode());
    }
    const output = tf.tanh(this.layers[this.layers.length - 1]);
    this.layers.forEach(layer => layer.dispose());
    return output;
  }

  dispose() {
    Object.values(this.variables).forEach(variable => variable.dispose());
  }
}

const loadPictModel = async (path) => {
  const checkpointLoader = new CheckpointLoaderPix2pix(path);
  const variables = await checkpointLoader.getAllVariables();
  return new PictModel(variables);
}

class Pix2pix {
  /**
   * Create a Pix2pix model.
   * @param {model} model - The path for a valid model.
   * @param {function} callback  - Optional. A function to run once the model has been loaded. If no callback is provided, it will return a promise that will be resolved once the model has loaded.
   */
  constructor(model, callback) {
    /**
     * Boolean to check if the model has loaded
     * @type {Promise<Pix2pix>}
     * @public
     */
    this.ready = callCallback(this.load(model), callback);
  }

  async load(path) {
    /**
     * @type {PictModel}
     */
    this.model = await loadPictModel(path);
    return this;
  }

  /**
   * Given an canvas element, applies image-to-image translation using the provided model. Returns an image.
   * @param {HTMLCanvasElement} inputElement
   * @param {function} cb - A function to run once the model has made the transfer. If no callback is provided, it will return a promise that will be resolved once the model has made the transfer.
   */
  async transfer(inputElement, cb) {
    return callCallback(this.transferInternal(inputElement), cb);
  }

  // TODO: return multi-format result using generatedImageResult.
  async transferInternal(inputElement) {
    await this.ready;
    const result = array3DToImage(tf.tidy(() => {
      const preprocessedInput = Pix2pix.preprocess(inputElement);
      const output = this.model.predict(preprocessedInput);
      return Pix2pix.deprocess(output);
    }));
    await tf.nextFrame();
    return result;
  }

  static preprocess(inputElement) {
    return tf.tidy(() => {
      const input = tf.browser.fromPixels(inputElement);
      // Convert from int [0,255] to float [0,1]
      const floatInput = tf.div(tf.cast(input, 'float32'), tf.scalar(255));
      // Convert to range [-1,1]
      return tf.sub(tf.mul(floatInput, tf.scalar(2)), tf.scalar(1));
    });
  }

  static deprocess(inputDeproc) {
    return tf.tidy(() => {
      // Convert from [-1,1] back to [0,1]
      return tf.div(tf.add(inputDeproc, tf.scalar(1)), tf.scalar(2));
    });
  }

  dispose() {
    this.model.dispose();
  }
}

const pix2pix = (model, callback) => {
  const instance = new Pix2pix(model, callback);
  return callback ? instance : instance.ready;
};

export default pix2pix;
