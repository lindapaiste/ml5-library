// Copyright (c) 2020 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
 * Handpose: Palm detector and hand-skeleton finger tracking in the browser
 * Ported and integrated from all the hard work by: https://github.com/tensorflow/tfjs-models/tree/master/handpose
 */

import * as handposeCore from "@tensorflow-models/handpose";
import { Prediction } from "@tensorflow-models/handpose/dist/pipeline";
import { ImageModel } from "../_rework/Image/types";

export type HandposeOptions = {
// passed to load()
  maxContinuousChecks?: number
  detectionConfidence?: number;
  iouThreshold?: number;
  scoreThreshold?: number;
// passed to predict()
  flipHorizontal?: boolean;
};

// Note: AnnotatedPrediction type is not exported from handpose :(
interface AnnotatedPrediction extends Prediction {
  annotations: {
    [key: string]: Array<[number, number, number]>;
  };
}

export default async function createHandposeModel(initialOptions: HandposeOptions = {}): Promise<ImageModel<AnnotatedPrediction[], HandposeOptions>> {
  let model = await handposeCore.load(initialOptions);
  return {
    name: 'Handpose',
    event: 'hand',
    detect: async (img, options) => {
      const flipHorizontal = options?.flipHorizontal || false;
      if (options) { // TODO: better way to figure out when to load/reload
        model = await handposeCore.load(options);
      }
      // @ts-ignore TODO
      return model.estimateHands(img, flipHorizontal);
    },
    defaultOptions: {}
  }
}

export const ml5handpose = createHandposeModel();
