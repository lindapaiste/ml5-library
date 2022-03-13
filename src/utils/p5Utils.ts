// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import type p5 from 'p5';

class P5Util {
  private m_p5Instance: Window | p5 | { p5: p5 } | undefined;

  constructor() {
    if (typeof window !== "undefined") {
      /**
       * Store the window as a private property regardless of whether p5 is present.
       * Can also set this property by calling method setP5Instance().
       * @property {Window | p5 | {p5: p5} | undefined} m_p5Instance
       * @private
       */
      this.m_p5Instance = window;
    }
  }

  /**
   * Set p5 instance globally in order to enable p5 features throughout ml5.
   * Call this function with the p5 instance when using p5 in instance mode.
   * @param {p5 | {p5: p5}} p5Instance
   */
  setP5Instance(p5Instance: p5 | { p5: p5 }) {
    this.m_p5Instance = p5Instance;
  }

  /**
   * Dynamic getter checks if p5 is loaded and will return undefined if p5 cannot be found,
   * or will return an object containing all of the global p5 properties.
   * It first checks if p5 is in the window, and then if it is in the p5 property of this.m_p5Instance.
   * @returns {p5 | undefined}
   */
  get p5Instance(): p5 | undefined {
    if (!this.m_p5Instance) return undefined;
    if (typeof (this.m_p5Instance as p5).loadImage === "function") {
      return this.m_p5Instance as p5;
    }

    if (typeof (this.m_p5Instance as { p5: p5 }).p5 !== 'undefined' &&
      // TODO: should it be .image instead of .Image?
      typeof (this.m_p5Instance as any).p5.Image !== 'undefined' &&
      typeof (this.m_p5Instance as any).p5.Image === 'function') {
      return (this.m_p5Instance as { p5: p5 }).p5;
    }
    return undefined;
  }

  /**
   * This function will check if the p5 is in the environment
   * Either it is in the p5Instance mode OR it is in the window
   * @returns {boolean} if it is in p5
   */
  checkP5(): this is this & { p5Instance: p5 } {
    return !!this.p5Instance;
  }

  /**
   * Convert a canvas to a Blob object.
   * @param {HTMLCanvasElement} inputCanvas
   * @returns {Promise<Blob>}
   */

  /* eslint class-methods-use-this: ["error", { "exceptMethods": ["getBlob"] }] */
  getBlob(inputCanvas: HTMLCanvasElement): Promise<Blob> {
    return new Promise((resolve, reject) => {
      inputCanvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Canvas could not be converted to Blob.'));
        }
      });
    });
  };

  /**
   * Load a p5.Image from a URL in an async way.
   * @param {string} url
   * @return {Promise<p5.Image>}
   */
  loadAsync(url: string): Promise<p5.Image> {
    const p5 = this.p5Instance;
    if (!p5) {
      return Promise.reject(new Error('No p5 instance found.'));
    }
    return new Promise((resolve, reject) => {
      p5.loadImage(url, (img: p5.Image) => {
        resolve(img);
      }, () => {
        reject(new Error(`Could not load image from url ${url}`));
      });
    });
  };

  /**
   * convert raw bytes to blob object
   * @param {number[] | Uint8ClampedArray | ArrayLike<number>} raws
   * @param {number} width
   * @param {number} height
   * @returns {Promise<Blob>}
   */
  async rawToBlob(raws: number[] | Uint8ClampedArray | ArrayLike<number>, width: number, height: number) {
    const arr = Array.from(raws);
    const canvas = document.createElement('canvas'); // Consider using offScreenCanvas when it is ready?
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Error accessing canvas 2D rendering context');
    }

    canvas.width = width;
    canvas.height = height;

    const imgData = ctx.createImageData(width, height);
    const { data } = imgData;

    for (let i = 0; i < width * height * 4; i += 1) data[i] = arr[i];
    ctx.putImageData(imgData, 0, 0);

    return this.getBlob(canvas);
  };

  /**
   * Convert Blob to P5.Image
   * @param {Blob} blob
   * Note: may want to reject instead of returning null.
   * @returns {Promise<p5.Image | null>}
   */
  async blobToP5Image(blob: Blob): Promise<p5.Image | null> {
    if (this.checkP5() && typeof URL !== "undefined") {
      return this.loadAsync(URL.createObjectURL(blob));
    }
    return null;
  };

}

const p5Utils = new P5Util();

export default p5Utils;
