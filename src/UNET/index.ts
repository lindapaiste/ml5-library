// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Image Classifier using pre-trained networks
*/

import * as tf from '@tensorflow/tfjs';
import callCallback, { ML5Callback } from '../utils/callcallback';
import {ArgSeparator} from "../utils/argSeparator";
import {generatedImageResult, groupedResult} from "../utils/GeneratedImage";
import { ImageArg, InputImage } from "../utils/handleArguments";
import {toTensor} from "../utils/imageConversion";
import {Tensor} from "@tensorflow/tfjs-core";
import {P5Image} from "../utils/p5Utils";


interface UNETOptions {
  modelPath?: string;
  imageSize?: number;
  returnTensors?: boolean;
}

const DEFAULTS: Required<UNETOptions> = {
  modelPath: 'https://raw.githubusercontent.com/zaidalyafeai/HostedModels/master/unet-128/model.json',
  imageSize: 128,
  returnTensors: true,
}

interface UNETSegmentation {
  segmentation: ImageData;
  blob: {
    featureMask: Blob;
    backgroundMask: Blob;
  },
  tensor?: {
    featureMask?: Tensor;
    backgroundMask?: Tensor;
  },
  raw: {
    featureMask: ImageData;
    backgroundMask: ImageData;
  },
  // returns if p5 is available
  featureMask: P5Image | HTMLCanvasElement;
  backgroundMask: P5Image | HTMLCanvasElement;
  mask: P5Image | HTMLCanvasElement;
}

class UNET {
  video?: HTMLVideoElement;
  config: Required<UNETOptions>;
  modelReady: boolean;
  isPredicting: boolean;
  ready: Promise<UNET>;
  model?: tf.LayersModel;
  /**
   * Create UNET class. 
   * @param {HTMLVideoElement | HTMLImageElement} video - The video or image to be used for segmentation.
   * @param {Object} options - Optional. A set of options.
   * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise 
   *    that will be resolved once the model has loaded.
   */
  constructor(video?: HTMLVideoElement, options?: UNETOptions, callback?: ML5Callback<UNET>) {
    this.modelReady = false;
    this.isPredicting = false;
    this.config = {
      ...DEFAULTS,
      ...options,
    };
    this.ready = callCallback(this.loadModel(), callback);
  }

  private async loadModel(): Promise<this> {
    this.model = await tf.loadLayersModel(this.config.modelPath);
    this.modelReady = true;
    return this;
  }

  public async segment(inputOrCallback?: ImageArg | ML5Callback<UNETSegmentation>, cb?: ML5Callback<UNETSegmentation>): Promise<UNETSegmentation> {
    const {image, callback} = new ArgSeparator(inputOrCallback, cb);
    return callCallback(this.segmentInternal(image), callback);
  }

  private async createMasks(imgToPredict: InputImage) {
    return tf.tidy(() => {
      // preprocess the input image
      const tfImage = toTensor(imgToPredict).toFloat();
      const resizedImg = tf.image.resizeBilinear(tfImage, [this.config.imageSize, this.config.imageSize]);
      let normTensor = resizedImg.div(tf.scalar(255));
      const batchedImage = normTensor.expandDims(0);
      // get the segmentation
      const pred = this.model!.predict(batchedImage) as tf.Tensor;

      // add back the alpha channel to the normalized input image
      const alpha = tf.ones([128, 128, 1]).tile([1,1,1])
      normTensor = normTensor.concat(alpha, 2)

      // get the feature mask
      // all of these operations are also used by the background mask
      const sharedBase = pred
          .squeeze([0])
          .tile([1, 1, 4])
          .sub(0.3)
          .sign()
          .relu();
      const backgroundMask = sharedBase
          .mul(normTensor);

      // get the background mask;
      const featureMask = sharedBase
          .neg()
          .add(1)
          .mul(normTensor);


      const alpha255 = tf.ones([128, 128, 1]).tile([1,1,1]).mul(255);
      let newpred = pred.squeeze([0]);
      newpred = tf.cast(newpred.tile([1,1,3]).sub(0.3).sign().relu().mul(255), 'int32')
      newpred = newpred.concat(alpha255, 2)

      return {
        featureMask,
        backgroundMask,
        segmentation: newpred
      };
    });
  }

  private async segmentInternal(imgToPredict: InputImage): Promise<UNETSegmentation> {
    // Wait for the model to be ready
    await this.ready;
    // skip asking for next frame if it's not video
    if (imgToPredict instanceof HTMLVideoElement) {
      await tf.nextFrame();
    }
    this.isPredicting = true;

    const {featureMask, backgroundMask, segmentation} = await this.createMasks(imgToPredict);

    this.isPredicting = false;

    // formatting the results into the expected response

    const maskRes = await generatedImageResult(segmentation, this.config);

    const grouped = await groupedResult({
      featureMask,
      backgroundMask
    });

    const {image, ...rest} = grouped;

    return {
      ...rest,
      ...image,
      segmentation: maskRes.raw,
      mask: maskRes.image
    }
  }
}

const uNet = (videoOr?: HTMLVideoElement | UNETOptions | ML5Callback<UNET>, optionsOr?: UNETOptions | ML5Callback<UNET>, cb?: ML5Callback<UNET>) => {
const {video, options, callback} = new ArgSeparator(videoOr, optionsOr, cb);
  return new UNET(video, options, callback);
};

export default uNet;
