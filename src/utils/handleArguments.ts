import * as tf from "@tensorflow/tfjs";
import { createCanvas, Image as cImage, ImageData as cImageData } from "canvas";
import type p5 from 'p5';
import { ML5Callback } from "./callcallback";

/**
 * @typedef ImageElement
 * @type {HTMLImageElement | HTMLCanvasElement | HTMLVideoElement}
 */
export type ImageElement = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement

export type PixelData = { data: Uint8Array; width: number; height: number }

/**
 * Standard input accepted by most TensorFlow models.
 * @typedef InputImage
 * @type {ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | tf.Tensor3D}
 */
export type InputImage = ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | tf.Tensor3D

/**
 * ML5 models accept all TensorFlow image inputs as well as p5 images and videos.
 * @typedef ImageArg
 * @type {InputImage | p5.Image | p5.Video | p5.Element}
 */
export type ImageArg = InputImage | p5.Image | p5.Element | p5.Graphics | cImage | cImageData | PixelData

export type VideoArg = HTMLVideoElement | p5.MediaElement;

/**
 * Check if a variable is an HTMLVideoElement.
 * @param {any} img
 * @returns {img is HTMLVideoElement}
 */
export const isVideo = (img: unknown): img is HTMLVideoElement => {
  // Must guard all instanceof checks on DOM elements in order to run in node.
  return typeof (HTMLVideoElement) !== 'undefined' &&
    img instanceof HTMLVideoElement;
}

/**
 * Check if a variable is an HTMLCanvasElement.
 * @param {any} img
 * @returns {img is HTMLCanvasElement}
 */
export const isCanvas = (img: unknown): img is HTMLCanvasElement => {
  return typeof (HTMLCanvasElement) !== 'undefined' &&
    img instanceof HTMLCanvasElement;
}

/**
 * Check if a variable is an HTMLImageElement.
 * @param {any} img
 * @returns {img is HTMLImageElement}
 */
export const isImg = (img: unknown): img is HTMLImageElement => {
  return typeof (HTMLImageElement) !== 'undefined' &&
    img instanceof HTMLImageElement;
}

/**
 * Check if a variable is a p5.Image or other p5.Element.
 * @param {p5.Element | p5.Image} img
 * @returns {img is p5.Element | p5.Image}
 */
export const isP5Image = (img: object): img is p5.Image | p5.Graphics | p5.Element => {
  return 'elt' in img || 'canvas' in img;
}

/**
 * Check if a variable is an instance of ImageData,
 * or a plain object with the same properties as ImageData.
 * This allows it to work in Node environments where ImageData is not defined.
 * @param {any} img
 * @returns {img is ImageData}
 */
export const isImageData = (img: any): img is ImageData => {
  if (typeof (ImageData) === 'undefined') {
    return (
      typeof img === 'object' &&
      // TODO: figure out TensorFlow issues with Uint8ClampedArray vs. Uint8Array
      (img.data instanceof Uint8ClampedArray || img.data instanceof Uint8Array) &&
      typeof img.width === 'number' &&
      typeof img.height === 'number'
    )
  }
  return img instanceof ImageData;
}

/**
 * Check if an unknown variable is a TensorFlow tensor with rank 3.
 * @param {any} img
 * @returns {img is tf.Tensor3D}
 */
export const isTensor3D = (img: unknown): img is tf.Tensor3D => {
  return img instanceof tf.Tensor && img.rank === 3;
}

/**
 * Check if an image is one of HTMLImageElement, HTMLCanvasElement, HTMLVideoElement
 * @param {any} img
 * @returns {img is ImageElement}
 */
export const isImageElement = <T = unknown>(img: T): img is T & ImageElement => {
  return !!img && (isCanvas(img) || isImg(img) || isVideo(img));
}

/**
 * Check that the provided image is an acceptable format and return it.
 * If it is a p5 Image, return the underlying HTML element.
 * Otherwise, return undefined.
 * Use overloads to refine the return type based on the argument type.
 * @param {any} img
 * @returns {ImageElement | null}
 */
export function getImageElement<T extends ImageElement>(img: T | { canvas: T } | { elt: T }): T
export function getImageElement<T extends ImageElement>(img: T | { canvas: T } | { elt: T } | unknown): T | undefined
export function getImageElement(img: unknown): ImageElement | undefined
export function getImageElement(img: any): ImageElement | undefined {
  if (isImageElement(img)) {
    return img;
  }
  if (typeof img === 'object') {
    if (isImageElement(img.canvas)) {
      return img.canvas;
    }
    if (isImageElement(img.elt)) {
      return img.elt;
    }
  }
  return undefined;
}

const convertImageData = ({ width, height, data }: PixelData | ImageData): tf.Tensor3D => {
  return tf.browser.fromPixels({ width, height, data: new Uint8Array(data) });
}

export const handlePolyfill = (img: any): tf.Tensor3D | null => {
  if (isImageData(img)) {
    return convertImageData(img)
  }
  if (img instanceof cImage) {
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, img.width, img.height);
    const data = ctx.getImageData(0, 0, img.width, img.height);
    return convertImageData(data);
  }
  return null;
}

export const toTensor = (img: InputImage): tf.Tensor3D => {
  if (isTensor3D(img)) {
    return img;
  }
  return tf.browser.fromPixels(img);
}


interface StandardArguments {
  /**
   * A model name or any other string argument.
   */
  string?: string;
  /**
   * Any numeric argument, such as number of classes, maximum iterations, accuracy, etc.
   */
  number?: number;
  /**
   * A callback function.
   */
  callback?: ML5Callback<any>;
  /**
   * Any object which is not a media object is assumed to be options.
   */
  options?: Record<string, any>;
  /**
   * Any array.
   */
  array?: any[];
  /**
   * Both video and audio-only elements will be assigned to the audio property.
   */
  audio?: HTMLMediaElement;
  /**
   * Video elements also get their own property.
   */
  video?: HTMLVideoElement;
  /**
   * Any video, image, or image data.
   */
  image?: InputImage;
}

/**
 * For methods which accept multiple optional arguments, a specific argument might be passed in multiple positions.
 * We can determine which argument is which by examining their values.
 *
 * Creates an object where each argument is assigned to a named property.
 * All properties are optional as arguments might be missing.
 */
class ArgHelper<Args extends any[]> implements StandardArguments {

  /**
   * Strict types here mean no errors when destructuring,
   * but makes assignment inside this class tougher.
   */
  image?: InputImage;
  video?: HTMLVideoElement;
  audio?: HTMLMediaElement;
  options?: Exclude<Extract<Args[number], object>, ImageArg | Function | any[]>;
  callback?: Extract<Args[number], Function>;
  string?: string;
  number?: number;
  array?: Extract<Args[number], any[]>;

  /**
   * Arguments used to CREATE an image-based model can be:
   *  - video: an HTMLVideoElement or p5 video.
   *  - options: an object of options specific to this model.
   *  - callback: a function to run once the model has been loaded. Called with arguments (error) or (error, result).
   *  - modelName: some models accept a model name or URL as an argument.
   *
   * Arguments used to CALL a method an image-based model can be:
   *  - image: an image or video element or an ImageData object.  Valid types: HTMLImageElement, HTMLCanvasElement,
   *    HTMLVideoElement, ImageData, p5 image, p5 video.
   *  - options: an object of options specific to this model.
   *  - callback: a function to run once the method has been completed.
   *
   * Expected to be provided in order modelName, video/image, options, callback with any omitted.
   * This function does not actually require any particular order.
   *
   * Later arguments will override earlier ones, so `this.video` should always be the first when providing arguments
   * from a class method call.
   *
   *  @param {any[]} [args]
   */
  constructor(...args: any[]) {
    args.forEach((arg) => this.addArg(arg));
  }

  /**
   * Can add arguments through the constructor or at any time after construction.
   *
   * @param {any} arg
   */
  addArg(arg: any) {
    // skip over falsey arguments and don't throw any error, assuming that these are omissions
    // do this check first to prevent accessing properties on null, which is an object
    if (arg === undefined || arg === null) {
      return;
    }
    switch (typeof arg) {
      case "string":
        this.string = arg;
        break;
      case "number":
        this.number = arg;
        break;
      case "function":
        this.callback = arg
        break;
      case "object": {
        if (isTensor3D(arg) || isImageData(arg)) {
          this.image = arg;
        }
        // Handle p5 object and HTML elements.
        const element = getImageElement(arg);
        if (element) {
          this.image = element;
          // Videos are also both images and audio.
          if (isVideo(element)) {
            this.audio = element
            this.video = element;
          }
        }
          // TODO: handle audio elements and p5.sound
        // Check for arrays
        else if (Array.isArray(arg)) {
          this.array = arg as any;
        }
        // All other objects are assumed to be options.
        else {
          this.options = arg;
        }
        break;
      }
      default:
        // Notify user about invalid arguments (would be ok to just skip)
        throw new Error("invalid argument"); // TODO: better message.
    }
  }

  /**
   * Check whether or not a given property has been set.
   *
   * @param {string & keyof StandardArguments} property
   * @returns {boolean}
   */
  has<K extends keyof StandardArguments>(property: K): this is this & Record<K, NonNullable<this[K]>> {
    return this[property] !== undefined;
  }

  /**
   * Check that an argument exists and throw an error if it doesn't.
   *
   * @param {string & keyof StandardArguments} property
   * @param {string} [message]
   * @return {this}
   */
  require<K extends keyof StandardArguments>(property: K, message?: string): this & Record<K, NonNullable<this[K]>> {
    if (this.has(property)) {
      return this;
    }
    throw new Error(message || `An argument for ${property} must be provided.`);
  }
}

/**
 * Export a chainable method instead of the class itself.
 */
export default function handleArguments<Args extends any[]>(...args: Args) {
  return new ArgHelper<Args>(...args);
};
