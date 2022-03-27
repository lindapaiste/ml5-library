// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tf from '@tensorflow/tfjs';
import {LayersModel, Tensor} from '@tensorflow/tfjs';
import {Classification, getTopKClassesFromTensor} from '../utils/gettopkclasses';
import DOODLENET_CLASSES from '../utils/DOODLENET_CLASSES';
import {ClassifierModel, ClassifierModelConfig} from "./darknet";
import {TfImageSource} from "../utils/imageUtilities";
import {toTensor} from "../utils/imageConversion";
import {Tensor3D} from "@tensorflow/tfjs-core";

const SETTINGS: ClassifierModelConfig = {
  url: 'https://cdn.jsdelivr.net/gh/ml5js/ml5-data-and-models@master/models/doodlenet/model.json',
  imgSize: 28,
  classes: DOODLENET_CLASSES,
};

function preProcess(img: TfImageSource, size: number) {
  const image = toTensor(img);
  const normalized: Tensor3D = tf.scalar(1).sub(image.toFloat().div(tf.scalar(255)));
  let resized = normalized;
  if (normalized.shape[0] !== size || normalized.shape[1] !== size) {
    resized = tf.image.resizeBilinear(normalized, [size, size]);
  }

  // TODO: why does doodlenet add a gray vector and darknet doesn't?
  const [r, g, b] = tf.split(resized, 3, 3);
  // Get average r,g,b color value and round to 0 or 1
  const gray = (r.add(g).add(b)).div(tf.scalar(3)).floor();
  // batched
  return gray.reshape([1, size, size, 1]);
}

/**
 * Taking the settings in the constructor with the hope of eventually combining with Darknet.
 */
export class Doodlenet implements ClassifierModel {
  constructor(public readonly model: LayersModel, private config: ClassifierModelConfig) {
  }

  async warmup(): Promise<void> {
    // Warmup the model.
    const result = tf.tidy(() => this.model.predict(
        // TODO: why is fourth argument 1 here and 3 in doodlnet?
        tf.zeros([1, this.config.imgSize, this.config.imgSize, 1])
    ) as Tensor);
    await result.data();
    result.dispose();
  }

  async classify(img: TfImageSource, topk: number = 10): Promise<Classification[]> {
    const logits = tf.tidy(() => {
      const imgData = preProcess(img, this.config.imgSize);
      return this.model.predict(imgData) as Tensor;
    });
    const classes = await getTopKClassesFromTensor(logits, topk, this.config.classes);
    logits.dispose();
    return classes;
  }
}

export const load = async (): Promise<ClassifierModel> => {
  const model = await tf.loadLayersModel(SETTINGS.url);
  const doodlenet = new Doodlenet(model, SETTINGS);
  await doodlenet.warmup();
  return doodlenet;
}
