// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tf from '@tensorflow/tfjs';
import {Dimensions, getHeight, getWidth, HasDimensions} from "./dimensions";
import {Tensor} from "@tensorflow/tfjs";
import {Tensor3D} from "@tensorflow/tfjs-core";

/**
 * A union of element types which can be used as sources
 * @typedef {(HTMLVideoElement | HTMLImageElement | HTMLCanvasElement)} MediaElement
 * // TODO: what about ImageBitmap? SVGImageElement?
 */
export type MediaElement = HTMLVideoElement | HTMLImageElement | HTMLCanvasElement;


/**
 * A union of element types which can be used as sources by TensorFlow.
 * Includes HTML elements as well as raw data.
 * @typedef {(HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | ImageData | tf.Tensor3D)} TfImageSource
 */
export type TfImageSource = MediaElement | tf.Tensor3D | ImageData;

/**
 * When using a p5 image as a source, expect to find a valid element on the `elt` or `canvas` properties
 * @typedef {Object} P5Element
 * @property {TfImageSource} [elt]
 * @property {HTMLCanvasElement} [canvas]
 */
export interface P5Element {
  elt: TfImageSource;
}
export interface P5Image {
  canvas: HTMLCanvasElement;
}

/**
 * A source can be a valid element or a p5 image which contains an element
 * @typedef {(TfImageSource | P5Element)} ImageArg
 */
export type ImageArg = TfImageSource | P5Element | P5Image;
/*export type ImageArg = TfImageSource | {
  elt: MediaElement;
} | {
  canvas: HTMLCanvasElement;
}*/

/**
 * Models with a video argument can accept a video element or p5.Element
 * @typedef {(HTMLVideoElement | {elt: HTMLVideoElement})} VideoArg
 */
export type VideoArg = HTMLVideoElement | {
  elt: HTMLVideoElement;
}

/**
 * Extend a data source by adding width and height properties
 * @typedef {Object} ImageWithSize
 * @extends {Dimensions}
 * @property {TfImageSource} data
 */
export interface ImageWithSize extends Dimensions {
  data: TfImageSource;
}
/**
 * Checks if a subject is an element that can be used as a data source
 *
 * @param subject
 * @return boolean - true if subject is one of HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData, tf.Tensor
 */
export const isInstanceOfSupportedElement = (subject: any): subject is TfImageSource => {
  return (subject instanceof HTMLVideoElement
      || subject instanceof HTMLImageElement
      || subject instanceof HTMLCanvasElement
      || subject instanceof ImageData
      || subject instanceof tf.Tensor)
}

/**
 * Helper function to extract a data source from a variety of formats.
 *
 * @param {*} subject
 *  - a supported element (HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData).
 *  - a p5.js image, which is an object with an `elt` or `canvas` property.
 *  - other types are accepted, but will return `undefined`
 *
 * @return {TfImageSource | undefined} returns a valid source, or undefined if no such source can be found
 */
export const extractImageElement = (subject: any): TfImageSource | undefined => {
  // return valid elements
  if (isInstanceOfSupportedElement(subject)) {
    return subject;
  }
  // Handle p5.js image
  else if (subject && typeof subject === 'object') {
    if (isInstanceOfSupportedElement(subject.elt)) {
      return subject.elt;
    } else if (isInstanceOfSupportedElement(subject.canvas)) {
      return subject.canvas;
    }
  }
  // will return undefined if no valid source
  return undefined;
}

/**
 * Helper function to validate a video argument
 *
 * @param {*} subject - an HTMLVideoElement or p5 video
 */
export const extractVideoElement = (subject: any): HTMLVideoElement | undefined => {
  if (subject instanceof HTMLVideoElement) {
    return  subject;
  } else if (typeof subject === "object" && subject.elt instanceof HTMLVideoElement) {
    return subject.elt;
  }
}

/**
 * Async function that waits for a video to be loaded.
 * @param video
 */
export const videoLoaded = async (video: HTMLVideoElement): Promise<void> => {
  if ( video.readyState >= 2 ) {
    return;
  }
  return new Promise((resolve, reject) => {
    // Fired when the first frame of the media has finished loading.
    video.onloadeddata = () => resolve();
    // Fired when the resource could not be loaded due to an error.
    video.onerror = () => reject(new Error(`Error loading media file ${video.src}`));
  });
}

/**
 * Resize video elements
 *
 * @param videoInput {HTMLCanvasElement}
 * @param size {number} width and height of the resized video
 * @param [callback] {function} function to be called when the video is played
 * @return {HTMLVideoElement}
 */
export const processVideo = (videoInput: HTMLCanvasElement, size: number, callback = () => {}): HTMLVideoElement => {
  const element = document.createElement('video');
  videoInput.onplay = () => {
    element.srcObject = videoInput.captureStream?.();
    element.width = size;
    element.height = size;
    element.autoplay = true;
    element.playsInline = true;
    element.muted = true;
    callback();
  };
  return element;
};

/**
 * Apply cropping to a TensorFlow image based on the `shape` property
 *
 * @param {tf.Tensor<tf.Rank.R3>} img
 * @return {tf.Tensor<tf.Rank.R3>}
 */
export const cropImage = (img: tf.Tensor<tf.Rank.R3>): tf.Tensor<tf.Rank.R3> => {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - (size / 2);
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - (size / 2);
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
};

// Static Method: image to tf tensor
export function imgToTensor(input: TfImageSource, size: [number, number]): Tensor3D {
  return tf.tidy(() => {
    let img = input instanceof Tensor ? input : tf.browser.fromPixels(input);
    if (size) {
      img = tf.image.resizeBilinear(img, size);
    }
    const croppedImage = cropImage(img);
    const batchedImage = croppedImage.expandDims(0);
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  });
}


/**
 * Create a new `canvas` element with the provided dimensions
 */
export const createSizedCanvas = (object: HasDimensions): HTMLCanvasElement => {
  const canvas = document.createElement('canvas');
  canvas.width = getWidth(object);
  canvas.height = getHeight(object);
  return canvas;
}

/**
 * Helper function throws an Error when ctx is null, so the returned value is guaranteed to not be null.
 * @param canvas
 * @throws
 */
export const getCtx = (canvas: HTMLCanvasElement): CanvasRenderingContext2D => {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error("Error extracting rendering context from canvas element");
  }
  return ctx;
}

/*
export {
  array3DToImage,
  processVideo,
  cropImage,
  imgToTensor,
  isInstanceOfSupportedElement,
  flipImage,
  imgToPixelArray
};
*/

// probably not needed
function dataURLtoBlob(dataurl: string): Blob {
  const arr = dataurl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);

  while (n) {
    u8arr[n] = bstr.charCodeAt(n);
    n -= 1;
  }
  return new Blob([u8arr], {
    type: mime
  });
}