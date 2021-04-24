// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tf from '@tensorflow/tfjs';
import {Graphics} from 'p5';
import p5Utils from './p5Utils';

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
  elt?: TfImageSource;
  canvas?: HTMLCanvasElement;
}

/**
 * A source can be a valid element or a p5 image which contains an element
 * @typedef {(TfImageSource | P5Element)} ImageArg
 */
export type ImageArg = TfImageSource | P5Element;
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
 * @typedef {Object} Dimensions
 * @property {number} width
 * @property {number} height
 */
export interface Dimensions {
  width: number;
  height: number;
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
 * extract an object with `width` and `height` properties
 *
 * @param {TfImageSource | Dimensions} subject
 * @return {Dimensions}
 */
export const dimensions = (subject: TfImageSource | Dimensions): Dimensions => {
  // note: technically this condition is unnecessary,
  // as other elements do not have `videoWidth` and will correctly fallback to `width`
  if (subject instanceof HTMLVideoElement) {
    return {
      width: subject.videoWidth ?? subject.width,
      height: subject.videoHeight ?? subject.height,
    }
  } else {
    return {
      width: subject.width,
      height: subject.height
    }
  }
}

/**
 * Helper function returns a `data` source along with `width` and `height`
 *
 * @param subject
 * @return {ImageWithSize}
 * @throws {Error}
 */
export const extractElementWithSize = (subject: any): ImageWithSize => {
  const element = extractImageElement(subject);
  if ( element ) {
    return {
      ...dimensions(element),
      data: element,
    }
  }
  throw new Error("Unsupported source");
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
 * Converts a tf to DOM img
 *
 * @param tensor {tf.Tensor<tf.Rank.R3>}
 * @return {HTMLImageElement}
 */
export const array3DToImage = (tensor: tf.Tensor<tf.Rank.R3>): HTMLImageElement => {
  const [imgHeight, imgWidth] = tensor.shape;
  const data = tensor.dataSync();
  const canvas = document.createElement('canvas');
  canvas.width = imgWidth;
  canvas.height = imgHeight;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < imgWidth * imgHeight; i += 1) {
    const j = i * 4;
    const k = i * 3;
    imageData.data[j + 0] = Math.floor(256 * data[k + 0]);
    imageData.data[j + 1] = Math.floor(256 * data[k + 1]);
    imageData.data[j + 2] = Math.floor(256 * data[k + 2]);
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);

  // Create img HTML element from canvas
  const dataUrl = canvas.toDataURL();
  const outputImg = document.createElement('img');
  outputImg.src = dataUrl;
  outputImg.width = imgWidth;
  outputImg.height = imgHeight;
  tensor.dispose();
  return outputImg;
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

export const flipImage = (img) => {
  // image image, bitmap, or canvas
  let imgWidth;
  let imgHeight;
  let inputImg;

  if (img instanceof HTMLImageElement ||
      img instanceof HTMLCanvasElement ||
      img instanceof HTMLVideoElement ||
      img instanceof ImageData) {
    inputImg = img;
  } else if (typeof img === 'object' &&
      (img.elt instanceof HTMLImageElement ||
          img.elt instanceof HTMLCanvasElement ||
          img.elt instanceof HTMLVideoElement ||
          img.elt instanceof ImageData)) {

    inputImg = img.elt; // Handle p5.js image
  } else if (typeof img === 'object' &&
      img.canvas instanceof HTMLCanvasElement) {
    inputImg = img.canvas; // Handle p5.js image
  } else {
    inputImg = img;
  }

  if (inputImg instanceof HTMLVideoElement) {
    // should be videoWidth, videoHeight?
    imgWidth = inputImg.width;
    imgHeight = inputImg.height;
  } else {
    imgWidth = inputImg.width;
    imgHeight = inputImg.height;
  }


  if (p5Utils.checkP5()) {
    const p5Canvas: Graphics = p5Utils.p5Instance.createGraphics(imgWidth, imgHeight);
    p5Canvas.push()
    p5Canvas.translate(imgWidth, 0);
    p5Canvas.scale(-1, 1);
    p5Canvas.image(img, 0, 0, imgWidth, imgHeight);
    p5Canvas.pop()

    return p5Canvas;
  }
  const canvas = document.createElement('canvas');
  canvas.width = imgWidth;
  canvas.height = imgHeight;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(inputImg, 0, 0, imgWidth, imgHeight);
  ctx.translate(imgWidth, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(canvas, imgWidth * -1, 0, imgWidth, imgHeight);
  return canvas;

}

// Static Method: image to tf tensor
export function imgToTensor(input, size = null) {
  return tf.tidy(() => {
    let img = tf.browser.fromPixels(input);
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
 *
 * @param {Dimensions} dimensions
 * @param {number} width
 * @param {number} height
 * @return HTMLCanvasElement
 */
export const createCanvas = ({width, height}: Dimensions): HTMLCanvasElement => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/**
 * Create a new `canvas` element and use `drawImage` to copy the contents from a source
 *
 * @param {CanvasImageSource} source
 * @return {CanvasRenderingContext2D}
 */
export const copyToCanvas = (source: CanvasImageSource & Dimensions): CanvasRenderingContext2D => {
  // TODO: should videos use videoWidth?
  const canvas = createCanvas(dimensions(source));

  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0, source.width, source.height);
  return ctx;
  // TODO: do I ever want to return the canvas?
}

/**
 * convert an element source to an array of pixel data
 *
 * @param {ImageArg} source
 * @return {Uint8ClampedArray}
 */
export function imgToPixelArray(source: ImageArg): Uint8ClampedArray {
  const data = extractElementWithSize(source);

  //TODO: throw error here or in parent?
  if ( ! data ) {
    throw new Error("Unsupported source");
  }

  // TODO: ImageData is valid as a source for most applications, but not here
  const ctx = copyToCanvas(source);
  const canvas = document.createElement('canvas');
  canvas.width = imgWidth;
  canvas.height = imgHeight;


  const ctx = canvas.getContext('2d');
  ctx.drawImage(inputImg, 0, 0, imgWidth, imgHeight);

  const imgData = ctx.getImageData(0,0, imgWidth, imgHeight)
  // note: previous version cast to number[] with `return Array.from(imgData.data)`
  return Array.from(imgData.data)
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