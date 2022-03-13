// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
 * BodyPix: Real-time Person Segmentation in the Browser
 * Ported and integrated from all the hard work by: https://github.com/tensorflow/tfjs-models/tree/master/body-pix
 */

import * as bp from "@tensorflow-models/body-pix";
import { BodyPix } from "@tensorflow-models/body-pix";
import { MobileNetMultiplier, OutputStride } from "@tensorflow-models/body-pix/dist/mobilenet";
import { PartSegmentation, PersonSegmentation } from "@tensorflow-models/body-pix/dist/types";
import * as tf from "@tensorflow/tfjs";
import { Tensor3D } from "@tensorflow/tfjs";
import type p5 from 'p5';
import { ImageModel } from "../_rework/Image/types";
import { RGB, toRgb } from "../utils/colorUtilities";
import { InputImage, toTensor } from "../utils/handleArguments";
import p5Utils from '../utils/p5Utils';
import BODYPIX_PALETTE, { BodyPixPalette } from './BODYPIX_PALETTE';

export interface BodyPixOptions {
  multiplier: MobileNetMultiplier;
  outputStride: OutputStride;
  segmentationThreshold: number;
  palette: BodyPixPalette<RGB | number[] | p5.Color>;
  returnTensors: boolean;
  withParts: boolean;
}

const DEFAULTS: BodyPixOptions = {
  "multiplier": 0.75,
  "outputStride": 16,
  "segmentationThreshold": 0.5,
  "palette": BODYPIX_PALETTE,
  "returnTensors": false,
  "withParts": true
}

interface ResultObj {
  raw: ImageData;
  tensor?: Tensor3D;
  image?: p5.Image | null;
}

// Note: is backwards from before, ie. res.personMask.tensor instead of res.tensor.personMask.
interface PersonSegmentationResult {
  personMask: ResultObj;
  backgroundMask: ResultObj;
  segmentation: PersonSegmentation;
}

interface PartSegmentationResult {
  personMask: ResultObj;
  backgroundMask: ResultObj;
  partMask: ResultObj;
  segmentation: PartSegmentation;
  // TODO: what is this?
  bodyParts: BodyPixPalette;
}

/**
 * Returns a bodyPartsSpec object
 * @param {Array} an array of [r,g,b] colors
 * @return {object} an object with the bodyParts by color and id
 */
function bodyPartsSpec(palette: BodyPixPalette<RGB | number[] | p5.Color>): BodyPixPalette {
  // Check if we're getting p5 colors, convert to an RGB array
  return Object.fromEntries(Object.keys(palette).map((part) => {
    const { color, id } = palette[part];
    return [part, { id, color: toRgb(color) }];
  }));
}

/**
 * Returns a p5Image
 * @param {Uint8ClampedArray} pixels
 * @param {number} width
 * @param {number} height
 * @return {Promise<p5.Image>}
 */
async function convertToP5Image(pixels: Uint8ClampedArray, width: number, height: number): Promise<p5.Image | null> {
  const blob = await p5Utils.rawToBlob(pixels, width, height);
  return await p5Utils.blobToP5Image(blob);
}

// TODO: try to combine part & person more

const segmentWithParts = async (model: BodyPix, img: InputImage, options: BodyPixOptions): Promise<PartSegmentationResult> => {
  const segmentation = await model.estimatePartSegmentation(img, options.outputStride, options.segmentationThreshold);

  // TODO: handle merging and/or validating of part colors.
  /*
  const result = colorOptions !== undefined || Object.keys(colorOptions).length >= 24 ? colorOptions : this.config.palette;

  if (p5 && result !== undefined && Object.keys(result).length >= 24) {
  */

  const bodyPartsMeta = bodyPartsSpec(options.palette);
  const colorsArray = Object.values(bodyPartsMeta).map(object => object.color)

  const result: PartSegmentationResult = {
    segmentation,
    bodyParts: bodyPartsMeta,
    backgroundMask: {
      // TODO: can covert int32 to uint8 if needed
      raw: bp.toMaskImageData(segmentation as any, true)
    },
    personMask: {
      raw: bp.toMaskImageData(segmentation as any, false)
    },
    partMask: {
      raw: bp.toColoredPartImageData(segmentation, colorsArray)
    }
  };

  const {
    personMask,
    backgroundMask,
    partMask,
  } = tf.tidy(() => {
    // create a tensor from the input image
    const shape: [number, number, number] = [segmentation.height, segmentation.width, 1];
    const alpha: tf.Tensor3D = tf.ones(shape).tile([1, 1, 1]).mul(255)
    const normTensor = toTensor(img).concat(alpha, 2)

    // create a tensor from the segmentation
    // multiply the segmentation and the inputImage
    const maskPersonTensor = tf.tensor(segmentation.data, shape).add(0.2).sign().relu().mul(normTensor).cast('int32') as tf.Tensor3D;
    const maskBackgroundTensor = tf.tensor(segmentation.data, shape).add(0.2).sign().neg().relu().mul(normTensor).cast('int32') as tf.Tensor3D;
    // TODO: handle removing background
    const partTensor = tf.tensor([...result.partMask.raw.data], [segmentation.height, segmentation.width, 4]).cast('int32') as tf.Tensor3D;

    return {
      personMask: maskPersonTensor,
      backgroundMask: maskBackgroundTensor,
      partMask: partTensor
    }
  })

  const personMaskPixels = await tf.browser.toPixels(personMask);
  const bgMaskPixels = await tf.browser.toPixels(backgroundMask);
  const partMaskPixels = await tf.browser.toPixels(partMask);

  // if p5 exists, convert to p5 image
  if (p5Utils.checkP5()) {
    result.personMask.image = await convertToP5Image(personMaskPixels, segmentation.width, segmentation.height)
    result.backgroundMask.image = await convertToP5Image(bgMaskPixels, segmentation.width, segmentation.height)
    result.partMask.image = await convertToP5Image(partMaskPixels, segmentation.width, segmentation.height)
  }

  // otherwise, return the pixels
  /*result.personMask = personMaskPixels;
  result.backgroundMask = bgMaskPixels;
  result.partMask = partMaskPixels;*/

  if (!options.returnTensors) {
    personMask.dispose();
    backgroundMask.dispose();
    partMask.dispose();
  } else {
    // return tensors
    result.personMask.tensor = personMask;
    result.backgroundMask.tensor = backgroundMask;
    result.partMask.tensor = partMask;
  }

  return result;
}

const segmentPerson = async (model: BodyPix, img: InputImage, options: BodyPixOptions): Promise<PersonSegmentationResult> => {

  const segmentation: PersonSegmentation = await model.estimatePersonSegmentation(img, options.outputStride, options.segmentationThreshold);

  const result: PersonSegmentationResult = {
    segmentation,
    backgroundMask: {
      raw: bp.toMaskImageData(segmentation, true),
    },
    personMask: {
      raw: bp.toMaskImageData(segmentation, false)
    }
  }

  const {
    personMask,
    backgroundMask
  } = tf.tidy(() => {
    // create a tensor from the input image
    const shape: [number, number, number] = [segmentation.height, segmentation.width, 1];
    const alpha: tf.Tensor3D = tf.ones(shape).tile([1, 1, 1]).mul(255)
    const normTensor = toTensor(img).concat(alpha, 2)

    // create a tensor from the segmentation
    // multiply the segmentation and the inputImage
    const maskPersonTensor = tf.tensor3d(segmentation.data, shape).neg().add(1).mul(normTensor).cast('int32') as tf.Tensor3D;
    const maskBackgroundTensor = tf.tensor3d(segmentation.data, shape).mul(normTensor).cast('int32') as tf.Tensor3D;

    return {
      personMask: maskPersonTensor,
      backgroundMask: maskBackgroundTensor,
    }
  })

  // if p5 exists, convert to p5 image
  // TODO
  /*
  const personMaskPixels = await tf.browser.toPixels(personMask);
  const bgMaskPixels = await tf.browser.toPixels(backgroundMask);
  if (p5Utils.checkP5()) {
    result.personMask = await this.convertToP5Image(personMaskPixels, segmentation.width, segmentation.height)
    result.backgroundMask = await this.convertToP5Image(bgMaskPixels, segmentation.width, segmentation.height)
  } else {
    // otherwise, return the pixels
    result.personMask = personMaskPixels;
    result.backgroundMask = bgMaskPixels;
  }*/

  if (!options.returnTensors) {
    personMask.dispose();
    backgroundMask.dispose();
  } else {
    result.personMask.tensor = personMask;
    result.backgroundMask.tensor = backgroundMask;
  }

  return result;
}

export default async function createBodypixModel(initialOptions: BodyPixOptions = DEFAULTS): Promise<ImageModel<PersonSegmentationResult | PartSegmentationResult, BodyPixOptions>> {
  let model = await bp.load(initialOptions.multiplier);
  return {
    name: 'Bodypix',
    event: 'bodyParts',
    segment: async (img, options): Promise<PersonSegmentationResult | PartSegmentationResult> => {
      if (options.multiplier && options.multiplier !== initialOptions.multiplier) {
        // reload model on change of multiplier
        model = await bp.load(options.multiplier);
      }
      // TODO: there are many methods here
      if (options.withParts) {
        return segmentWithParts(model, img, options);
      } else {
        return segmentPerson(model, img, options);
      }
    },
    defaultOptions: DEFAULTS
  }
}
