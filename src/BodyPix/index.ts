// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
 * BodyPix: Real-time Person Segmentation in the Browser
 * Ported and integrated from all the hard work by: https://github.com/tensorflow/tfjs-models/tree/master/body-pix
 */

import * as tf from '@tensorflow/tfjs';
import * as bp from '@tensorflow-models/body-pix';
import callCallback, { ML5Callback } from '../utils/callcallback';
import { InputImage } from "../utils/handleArguments";
import p5Utils, {P5Image} from '../utils/p5Utils';
import BODYPIX_PALETTE from './BODYPIX_PALETTE';
import {MobileNetMultiplier, OutputStride} from "@tensorflow-models/body-pix/dist/mobilenet";
import {ArgSeparator} from "../utils/argSeparator";
import {Tensor3D} from "@tensorflow/tfjs-core";
import { isP5Color, P5Color, p5Color2RGB, RGB } from "../utils/colorUtilities";
import {toTensor} from "../utils/imageConversion";
import {generatedImageResult} from "../utils/GeneratedImage";

type BodyPalette<Color> = Record<string, { id: number; color: Color }>

interface BodyPixOptions {
  multiplier: MobileNetMultiplier;
  outputStride: OutputStride;
  segmentationThreshold: number;
  returnTensors: boolean;
  // TODO: accept input as just colors without ids. The user should not be able to change the ids!
  palette: BodyPalette<RGB | P5Color>;
}

const DEFAULTS: BodyPixOptions = {
  "multiplier": 0.75,
  "outputStride": 16,
  "segmentationThreshold": 0.5,
  "palette": BODYPIX_PALETTE,
  "returnTensors": false,
}

type Result<Keys extends string> = {
  segmentation: Uint8ClampedArray;
  raw: Record<Keys, ImageData>;
  tensor?: Record<Keys, Tensor3D>;
} & Record<Keys, P5Image | Uint8ClampedArray>;

type KeysWithoutParts = "personMask" | "backgroundMask";
type KeysWithParts = KeysWithoutParts | "partMask";

export type SegmentResult = Result<KeysWithoutParts>;

export type SegmentWithPartsResult = Result<KeysWithParts> & {
  bodyParts: BodyPalette<RGB>;
}

class BodyPix {
  video: HTMLVideoElement | null;
  config: BodyPixOptions;
  modelReady: boolean;
  modelPath: string;
  model?: bp.BodyPix;
  ready: Promise<BodyPix>;

  /**
   * Create BodyPix.
   * @param {HTMLVideoElement} video - An HTMLVideoElement.
   * @param {object} options - An object with options.
   * @param {function} callback - A callback to be called when the model is ready.
   */
  constructor(video?: HTMLVideoElement, options: Partial<BodyPixOptions> = {}, callback?: Callback<BodyPix>) {
    this.video = video || null;
    this.modelReady = false;
    this.modelPath = ''
    this.config = {
      ...options,
      ...DEFAULTS
    }
    this.ready = callCallback(this.loadModel(), callback);
  }

  /**
   * Load the model and set it to this.model
   * @return {this} the BodyPix model.
   */
  private async loadModel() {
    this.model = await bp.load(this.config.multiplier);
    this.modelReady = true;
    return this;
  }

  /**
   * Returns a p5Image
   * @param {*} tfBrowserPixelImage
   */
  async convertToP5Image(tfBrowserPixelImage, segmentationWidth, segmentationHeight) {
    const blob1 = await p5Utils.rawToBlob(tfBrowserPixelImage, segmentationWidth, segmentationHeight);
    return await p5Utils.blobToP5Image(blob1)
  }

  /**
   * Returns a bodyPartsSpec object
   * @param {Array} colorOptions an array of [r,g,b] colors
   * @return {object} an object with the bodyParts by color and id
   */
  private bodyPartsSpec(colorOptions: BodyPalette<RGB | P5Color>): BodyPalette<RGB> {
    // using keys form the default ensures that nothing extra is added and that all parts are present
    return Object.keys(BODYPIX_PALETTE).reduce((palette: BodyPalette<RGB>, part) => {
      const {color} = colorOptions[part] || this.config.palette[part] || BODYPIX_PALETTE[part];
      // use the ids from the default palette to ensure that they are not overwritten incorrectly.
      const id = BODYPIX_PALETTE[part].id;
      return {
        ...palette,
        [part]: {
          id,
          color: isP5Color(color) ? p5Color2RGB(color) : color,
        }
      }
    }, {});
    // could sort by id, but it's not needed because the defaults are in order
  }

  /**
   * Segments the image with partSegmentation, return result object
   * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} imgToSegment -
   *    takes any of the following params
   * @param {object} segmentationOptions - config params for the segmentation
   *    includes outputStride, segmentationThreshold
   * @return {Object} a result object with image, raw, bodyParts
   */
  public async segmentWithPartsInternal(imgToSegment: InputImage, segmentationOptions?: Partial<BodyPixOptions>) {
    // estimatePartSegmentation
    await this.ready;
    await tf.nextFrame();

    if (this.video && this.video.readyState === 0) {
      await new Promise<void>(resolve => {
        this.video.onloadeddata = () => resolve();
      });
    }

    const {palette, outputStride, segmentationThreshold} = {...this.config, ...segmentationOptions};

    const segmentation = await this.model!.estimatePartSegmentation(imgToSegment, outputStride, segmentationThreshold);

    const bodyPartsMeta = this.bodyPartsSpec(palette);
    const colorsArray = Object.values(bodyPartsMeta).map(obj => obj.color);

    const result = {
      segmentation,
      raw: {
        personMask: null,
        backgroundMask: null,
        partMask: null
      },
      tensor: {
        personMask: null,
        backgroundMask: null,
        partMask: null,
      },
      personMask: null,
      backgroundMask: null,
      partMask: null,
      bodyParts: bodyPartsMeta
    };
    const rawBackgroundMask = bp.toMaskImageData(segmentation, true);
    const rawPersonMask = bp.toMaskImageData(segmentation, false);
    const rawPartMask = bp.toColoredPartImageData(segmentation, colorsArray);

    const {
      personMask,
      backgroundMask,
      partMask,
    } = tf.tidy(() => {
      // create a tensor from the input image
      const alpha = tf.ones([segmentation.height, segmentation.width, 1]).tile([1, 1, 1]).mul(255);
      const normTensor = toTensor(imgToSegment).concat(alpha, 2);

      // create a tensor from the segmentation
      let maskPersonTensor = tf.tensor(segmentation.data, [segmentation.height, segmentation.width, 1]);
      let maskBackgroundTensor = tf.tensor(segmentation.data, [segmentation.height, segmentation.width, 1]);
      let partTensor = tf.tensor([...rawPartMask.data], [segmentation.height, segmentation.width, 4]);

      // multiply the segmentation and the inputImage
      maskPersonTensor = tf.cast(maskPersonTensor.add(0.2).sign().relu().mul(normTensor), 'int32')
      maskBackgroundTensor = tf.cast(maskBackgroundTensor.add(0.2).sign().neg().relu().mul(normTensor), 'int32')
      // TODO: handle removing background
      partTensor = tf.cast(partTensor, 'int32')

      return {
        personMask: maskPersonTensor,
        backgroundMask: maskBackgroundTensor,
        partMask: partTensor
      }
    })

    const person = generatedImageResult(personMask);
    const background = generatedImageResult(backgroundMask);
    const parts = generatedImageResult(partMask);

    // TODO: IMPORTANT: For backwards compatibility, return the pixels array instead of the canvas when no p5

    const personMaskPixels = await tf.browser.toPixels(personMask);
    const bgMaskPixels = await tf.browser.toPixels(backgroundMask);
    const partMaskPixels = await tf.browser.toPixels(partMask);

    // otherwise, return the pixels
    result.personMask = personMaskPixels;
    result.backgroundMask = bgMaskPixels;
    result.partMask = partMaskPixels;

    // if p5 exists, convert to p5 image
    if (p5Utils.checkP5()) {
      result.personMask = await this.convertToP5Image(personMaskPixels, segmentation.width, segmentation.height)
      result.backgroundMask = await this.convertToP5Image(bgMaskPixels, segmentation.width, segmentation.height)
      result.partMask = await this.convertToP5Image(partMaskPixels, segmentation.width, segmentation.height)
    }

    if (!this.config.returnTensors) {
      personMask.dispose();
      backgroundMask.dispose();
      partMask.dispose();
    } else {
      // return tensors
      result.tensor.personMask = personMask;
      result.tensor.backgroundMask = backgroundMask;
      result.tensor.partMask = partMask;
    }

    return result;

  }

  /**
   * Segments the image with partSegmentation
   * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} optionsOrCallback -
   *    takes any of the following params
   * @param {object} configOrCallback - config params for the segmentation
   *    includes palette, outputStride, segmentationThreshold
   * @param {function} cb - a callback function that handles the results of the function.
   * @return {function} a promise or the results of a given callback, cb.
   */
  public async segmentWithParts(optionsOrCallback, configOrCallback, cb) {
    const {image, options, callback} = new ArgSeparator(this.video, optionsOrCallback, configOrCallback, cb);
    if ( ! image ) { // Handle unsupported input
      throw new Error(
          'No input image provided. If you want to classify a video, pass the video element in the constructor.',
      );
    }

    return callCallback(this.segmentWithPartsInternal(image, options), callback);
  }

  /**
   * Segments the image with personSegmentation, return result object
   * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} imgToSegment -
   *    takes any of the following params
   * @param {object} segmentationOptions - config params for the segmentation
   *    includes outputStride, segmentationThreshold
   * @return {Object} a result object with maskBackground, maskPerson, raw
   */
  async segmentInternal(imgToSegment, segmentationOptions) {

    await this.ready;
    await tf.nextFrame();

    if (this.video && this.video.readyState === 0) {
      await new Promise<void>(resolve => {
        this.video.onloadeddata = () => resolve();
      });
    }

    this.config.outputStride = segmentationOptions.outputStride || this.config.outputStride;
    this.config.segmentationThreshold = segmentationOptions.segmentationThreshold || this.config.segmentationThreshold;

    const segmentation = await this.model.estimatePersonSegmentation(imgToSegment, this.config.outputStride, this.config.segmentationThreshold)

    const result = {
      segmentation,
      raw: {
        personMask: null,
        backgroundMask: null,
      },
      tensor: {
        personMask: null,
        backgroundMask: null,
      },
      personMask: null,
      backgroundMask: null,
    };
    result.raw.backgroundMask = bp.toMaskImageData(segmentation, true);
    result.raw.personMask = bp.toMaskImageData(segmentation, false);

    // TODO: consider returning the canvas with the bp.drawMask()
    // const bgMaskCanvas = document.createElement('canvas');
    // bgMaskCanvas.width = segmentation.width;
    // bgMaskCanvas.height = segmentation.height;
    // bp.drawMask(bgMaskCanvas, imgToSegment, result.maskBackground, 1, 3, false);

    // const featureMaskCanvas = document.createElement('canvas');
    // featureMaskCanvas.width = segmentation.width;
    // featureMaskCanvas.height = segmentation.height;
    // bp.drawMask(featureMaskCanvas, imgToSegment, result.maskPerson, 1, 3, false);

    // result.backgroundMask = bgMaskCanvas;
    // result.featureMask = featureMaskCanvas;

    const {
      personMask,
      backgroundMask
    } = tf.tidy(() => {
      let normTensor = tf.browser.fromPixels(imgToSegment);
      // create a tensor from the input image
      const alpha = tf.ones([segmentation.height, segmentation.width, 1]).tile([1, 1, 1]).mul(255)
      normTensor = normTensor.concat(alpha, 2)
      // normTensor.print();


      // TODO: can combine these next steps by chaining .cast(), but need to make sure it's the same
      // create a tensor from the segmentation
      let maskPersonTensor = tf.tensor(segmentation.data, [segmentation.height, segmentation.width, 1]);
      let maskBackgroundTensor = tf.tensor(segmentation.data, [segmentation.height, segmentation.width, 1]);

      // multiply the segmentation and the inputImage
      maskPersonTensor = tf.cast(maskPersonTensor.neg().add(1).mul(normTensor), 'int32')
      maskBackgroundTensor = tf.cast(maskBackgroundTensor.mul(normTensor), 'int32')

      return {
        personMask: maskPersonTensor,
        backgroundMask: maskBackgroundTensor,
      }
    })

    const personMaskPixels = await tf.browser.toPixels(personMask);
    const bgMaskPixels = await tf.browser.toPixels(backgroundMask);

    // if p5 exists, convert to p5 image
    if (p5Utils.checkP5()) {
      result.personMask = await this.convertToP5Image(personMaskPixels, segmentation.width, segmentation.height)
      result.backgroundMask = await this.convertToP5Image(bgMaskPixels, segmentation.width, segmentation.height)
    } else {
      // otherwise, return the pixels
      result.personMask = personMaskPixels;
      result.backgroundMask = bgMaskPixels;
    }

    if (!this.config.returnTensors) {
      personMask.dispose();
      backgroundMask.dispose();
    } else {
      result.tensor.personMask = personMask;
      result.tensor.backgroundMask = backgroundMask;
    }


    return result;

  }

  /**
   * Segments the image with personSegmentation
   * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} optionsOrCallback -
   *    takes any of the following params
   * @param {object} configOrCallback - config params for the segmentation
   *    includes outputStride, segmentationThreshold
   * @param {function} cb - a callback function that handles the results of the function.
   * @return {function} a promise or the results of a given callback, cb.
   */
  async segment(optionsOrCallback?: TfImageSource, configOrCallback?: BodyPixOptions, cb?: ML5Callback<any>) {
    const {image, options, callback} = ArgSeparator.from(this.video, optionsOrCallback, configOrCallback, cb)
        .require(
            'image',
            'No input image provided. If you want to classify a video, pass the video element in the constructor.'
        );
    return callCallback(this.segmentInternal(image, options), callback);
  }

}

const bodyPix = (videoOrOptionsOrCallback?: HTMLVideoElement | BodyPixOptions | ML5Callback<BodyPix>, optionsOrCallback?: BodyPixOptions | ML5Callback<BodyPix>, cb?: ML5Callback<BodyPix>) => {
  const {video, options, callback} = new ArgSeparator(videoOrOptionsOrCallback, optionsOrCallback, cb);

  const instance = new BodyPix(video, options, callback);
  return callback ? instance : instance.ready;
}

export default bodyPix;
