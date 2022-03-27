// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
DCGAN
This version is based on alantian's TensorFlow.js implementation: https://github.com/alantian/ganshowcase
*/

import * as tf from '@tensorflow/tfjs';
import callCallback, {Callback} from '../utils/callcallback';
import modelLoader, {isAbsoluteURL} from "../utils/modelLoader";
import {GeneratedImageResult, generatedImageResult} from "../utils/GeneratedImage";
import {ArgSeparator} from "../utils/argSeparator";
import {loadFile} from "../utils/io";
// Default pre-trained face model

// const DEFAULT = {
//     "description": "DCGAN, human faces, 64x64",
//     "model": "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/master/models/dcgan/face/model.json",
//     "modelSize": 64,
//     "modelLatentDim": 128
// }

/**
 * @typedef {Object} DCGANOptions
 * @property {boolean} returnTensors
 */
interface DCGANOptions {
    returnTensors?: boolean;
}

interface ModelInfo {
    modelLatentDim: number;
    model: string;
}

class DCGANBase {
    ready: Promise<DCGANBase>;
    modelReady: boolean;
    model?: tf.LayersModel;
    modelInfo?: ModelInfo;
    config: Required<DCGANOptions>;

    /**
     * Create an DCGAN.
     * @param {string} modelPath - The name of the model to use.
     * @param {DCGANOptions} [options]
     * @param {function} [callback] - A callback to be called when the model is ready.
     */
    constructor(modelPath: string, options: DCGANOptions = {}, callback?: Callback<DCGANBase>) {
        this.modelReady = false;
        this.config = {
            returnTensors: options.returnTensors || false,
        }
        this.ready = callCallback(this.loadModel(modelPath), callback);
    }

    /**
     * Load the model and set it to this.model
     * @param {string} manifestPath
     * @return {this} the dcgan.
     */
    private async loadModel(manifestPath: string): Promise<this> {
        // loads the manifest.json, which contains the relative path to the model.json.
        this.modelInfo = await loadFile<ModelInfo>(manifestPath);

        // maybe append the model path to the location of the manifest.
        const manifestLoader = modelLoader(manifestPath);
        const modelPath = this.modelInfo.model;
        const modelJsonPath = isAbsoluteURL(modelPath) ? modelPath : manifestLoader.fileInDirectory(modelPath);

        this.model = await tf.loadLayersModel(modelJsonPath);
        this.modelReady = true;
        return this;
    }

    /**
     * Computes what will become the image tensor
     * @param {number} latentDim - the number of latent dimensions to pass through
     * @param {object} latentVector - an array containing the latent vector; otherwise use random vector
     * @return {object} a tensor
     */
    private async compute(latentDim: number, latentVector?: number[]): Promise<tf.Tensor3D> {
        await this.ready;
        return tf.tidy(() => {
            let z;
            if (Array.isArray(latentVector)) {
                const buffer = tf.buffer([1, latentDim]);
                for (let count = 0; count < latentDim; count += 1) {
                    buffer.set(latentVector[count], 0, count);
                }
                z = buffer.toTensor();
            } else {
                z = tf.randomNormal([1, latentDim]);
            }
            // TBD: should model be a parameter to compute or is it ok to reference this.model here?
            return (this.model!.predict(z) as tf.Tensor).squeeze().transpose([1, 2, 0])
                .div(tf.scalar(2)).add<tf.Tensor3D>(tf.scalar(0.5));
        });
    }

    /**
     * Generates a new image
     * @param {function} callback - a callback function handle the results of generate
     * @param {object} latentVector - an array containing the latent vector; otherwise use random vector
     * @return {object} a promise or the result of the callback function.
     */
    public async generate(callback: Callback<GeneratedImageResult>, latentVector?: number[]): Promise<GeneratedImageResult> {
        await this.ready;
        return callCallback(this.generateInternal(latentVector), callback);
    }

    /**
     * Takes the tensor from compute() and returns an object of the generate image data
     * @param {object} latentVector - an array containing the latent vector; otherwise use random vector
     * @return {object} includes blob, raw, and tensor. if P5 exists, then a p5Image
     */
    private async generateInternal(latentVector?: number[]): Promise<GeneratedImageResult> {
        const {modelLatentDim} = this.modelInfo!;
        const imageTensor = await this.compute(modelLatentDim, latentVector);

        return generatedImageResult(imageTensor, this.config);
    }

}

const DCGAN = (modelPath: string, optionsOrCb?: DCGANOptions | Callback<DCGANBase>, cb?: Callback<DCGANBase>) => {
    const {callback, options, string} = new ArgSeparator(modelPath, optionsOrCb, cb);

    if (! string) {
        throw new Error(`Please specify a path to a "manifest.json" file: \n
         "models/face/manifest.json" \n\n
         This "manifest.json" file should include:\n
         {
            "description": "DCGAN, human faces, 64x64",
            "model": "https://raw.githubusercontent.com/viztopia/ml5dcgan/master/model/model.json", // "https://github.com/viztopia/ml5dcgan/blob/master/model/model.json",
            "modelSize": 64,
            "modelLatentDim": 128 
         }
         `);
    }

    const instance = new DCGANBase(string, options, callback);
    return callback ? instance : instance.ready;
}

export default DCGAN;
