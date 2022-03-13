// Copyright (c) 2020 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import { env } from "@tensorflow/tfjs-core";
import { Image } from "canvas";
import { ImageModelRunner } from "../_rework/Image/Image";
import { asyncLoadImage } from "../utils/testingUtils";

const HANDPOSE_IMG = 'https://i.imgur.com/EZXOjqh.jpg';

describe('Handpose', () => {
  const runner = new ImageModelRunner();
  let testImage: Image;

  beforeAll(async () => {
    jest.setTimeout(10000);
    env().set('IS_BROWSER', false);
  });

  it('detects hands in an image', async () => {
    testImage = await asyncLoadImage(HANDPOSE_IMG);
    const handPredictions = await runner.getMedia(testImage).detect('hand');
    expect(handPredictions).not.toHaveLength(0);
    expect(handPredictions[0].landmarks).toBeDefined();
    expect(handPredictions[0].landmarks).toEqual(5);
  });
});
