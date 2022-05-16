// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Image Classifier using pre-trained networks
*/

import * as tf from '@tensorflow/tfjs';
import callCallback, { ML5Callback } from '../utils/callcallback';
import constructorOverloads from '../utils/constructorOverloads';
import generatedImageResult from '../utils/generatedImageResult';
import handleArguments from "../utils/handleArguments";

const DEFAULTS = {
  modelPath: 'https://raw.githubusercontent.com/zaidalyafeai/HostedModels/master/unet-128/model.json',
  imageSize: 128,
  returnTensors: false,
}

interface UnetOptions {
  modelPath: string;
  imageSize: number;
  returnTensors: boolean;
}

class UNET {
  public ready: Promise<UNET>;
  public modelReady: boolean;
  public isPredicting: boolean;
  public config: UnetOptions;
  public model?: tf.LayersModel;
  public video?: HTMLVideoElement;

  /**
   * Create UNET class. 
   * @param {HTMLVideoElement | HTMLImageElement} video - The video or image to be used for segmentation.
   * @param {Object} options - Optional. A set of options.
   * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise 
   *    that will be resolved once the model has loaded.
   */
  constructor(video?: HTMLVideoElement, options?: Partial<UnetOptions>, callback?: ML5Callback<UNET>) {
    this.modelReady = false;
    this.isPredicting = false;
    this.config = {
      modelPath: typeof options.modelPath !== 'undefined' ? options.modelPath : DEFAULTS.modelPath,
      imageSize: typeof options.imageSize !== 'undefined' ? options.imageSize : DEFAULTS.imageSize,
      returnTensors: typeof options.returnTensors !== 'undefined' ? options.returnTensors : DEFAULTS.returnTensors,
    };
    this.video = video;
    this.ready = callCallback(this.loadModel(), callback);
  }

  async loadModel() {
    this.model = await tf.loadLayersModel(this.config.modelPath);
    this.modelReady = true;
    return this;
  }

  async segment(inputOrCallback, cb) {
    const { image, callback } = handleArguments(this.video, inputOrCallback, cb);
    await this.ready;
    return callCallback(this.segmentInternal(image), callback);
  }

  async segmentInternal(imgToPredict) {
    // Wait for the model to be ready
    await this.ready;
    // skip asking for next frame if it's not video
    if (imgToPredict instanceof HTMLVideoElement) {
      await tf.nextFrame();
    }
    this.isPredicting = true;

    const {
      featureMask,
      backgroundMask,
      segmentation
    } = tf.tidy(() => {
      // preprocess the input image
      const tfImage = tf.browser.fromPixels(imgToPredict).toFloat();
      const resizedImg = tf.image.resizeBilinear(tfImage, [this.config.imageSize, this.config.imageSize]);
      let normTensor = resizedImg.div(tf.scalar(255));
      const batchedImage = normTensor.expandDims(0);
      // get the segmentation
      const pred = this.model.predict(batchedImage) as tf.Tensor;
      
      // add back the alpha channel to the normalized input image
      const alpha = tf.ones([128, 128, 1]).tile([1,1,1])
      normTensor = normTensor.concat(alpha, 2)

      // TODO: optimize these redundancies below, e.g. repetitive squeeze() etc
      // get the background mask;
      let maskBackgroundInternal = pred.squeeze([0]);
      maskBackgroundInternal = maskBackgroundInternal.tile([1, 1, 4]);
      maskBackgroundInternal = maskBackgroundInternal.sub(0.3).sign().relu().neg().add(1);
      const featureMaskInternal = maskBackgroundInternal.mul(normTensor);

      // get the feature mask;
      let maskFeature = pred.squeeze([0]);
      maskFeature = maskFeature.tile([1, 1, 4]);
      maskFeature = maskFeature.sub(0.3).sign().relu();
      const backgroundMaskInternal = maskFeature.mul(normTensor);

      const alpha255 = tf.ones([128, 128, 1]).tile([1,1,1]).mul(255);
      let newpred = pred.squeeze([0]);
      newpred = tf.cast(newpred.tile([1,1,3]).sub(0.3).sign().relu().mul(255), 'int32') 
      newpred = newpred.concat(alpha255, 2)

      return {
        featureMask: featureMaskInternal as tf.Tensor3D,
        backgroundMask: backgroundMaskInternal as tf.Tensor3D,
        segmentation: newpred as tf.Tensor3D
      };
    });

    this.isPredicting = false;

    const maskFeat = await generatedImageResult(featureMask, this.config);
    const maskBg = await generatedImageResult(backgroundMask, this.config);
    const mask = await generatedImageResult(segmentation, this.config);

    return {
      segmentation: mask.raw,
      blob: {
        featureMask: maskFeat.blob,
        backgroundMask: maskBg.blob
      },
      tensor: {
        featureMask: maskFeat.tensor,
        backgroundMask: maskBg.tensor,
      },
      raw: {
        featureMask: maskFeat.raw,
        backgroundMask: maskBg.raw
      },
      // returns if p5 is available
      featureMask: maskFeat.image,
      backgroundMask: maskBg.image,
      mask: mask.image
    };
  }
}

const uNet = constructorOverloads.videoOptionsCallback(UNET);

export default uNet;
