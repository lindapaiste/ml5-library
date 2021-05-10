// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
import * as p5 from "p5";
import {p5InstanceExtensions} from "p5";

/**
 * Certain properties, such as the underlying `canvas`, aren't documented because they are considered private
 */
export interface P5Image extends p5.Image {
    canvas: HTMLCanvasElement;
    imageData: ImageData;
}

export interface P5Element<T> extends p5.Element {
    elt: T;
}

class P5Util {
    private m_p5Instance: (Partial<p5InstanceExtensions> & { p5?: p5 });

    constructor() {
        this.m_p5Instance = window;
    }

    /**
     * Set p5 instance globally.
     * @param {Object} p5Instance
     */
    setP5Instance(p5Instance: Partial<p5InstanceExtensions> & { p5?: p5 }) {
        this.m_p5Instance = p5Instance;
    }

    /**
     * This getter will return p5, checking first if it is in
     * the window and next if it is in the p5 property of this.m_p5Instance
     * @returns {boolean} if it is in p5
     */
    get p5Instance() {
        if (typeof this.m_p5Instance !== "undefined" &&
            typeof this.m_p5Instance.loadImage === "function") return this.m_p5Instance;

        if (typeof this.m_p5Instance.p5 !== 'undefined' &&
            typeof this.m_p5Instance.p5.Image !== 'undefined' &&
            typeof this.m_p5Instance.p5.Image === 'function') return this.m_p5Instance.p5;
        return undefined;
    }

    /**
     * This function will check if the p5 is in the environment
     * Either it is in the p5Instance mode OR it is in the window
     * @returns {boolean} if it is in p5
     */
    checkP5(): boolean {
        return !!this.p5Instance;
    }

    /**
     * Convert a canvas to Blob
     * @param {HTMLCanvasElement} inputCanvas
     * @returns {Blob} blob object
     */

    /* eslint class-methods-use-this: ["error", { "exceptMethods": ["toBlob"] }] */
    getBlob(inputCanvas: HTMLCanvasElement): Promise<Blob> {
        return new Promise((resolve) => {
            inputCanvas.toBlob((blob) => {
                resolve(blob);
            });
        });
    };

    /**
     * Load image in async way from a URL string.
     * URL can be an image src or a data src.
     * @param {string} url
     * @return {Promise<p5.Image>}
     */
    async loadAsync(url: string): Promise<P5Image> {
        return new Promise((resolve, reject) => {
            const p5 = this.p5Instance;
            if (!p5 || !p5.loadImage) {
                reject(new Error("p5 not loaded"));
            } else {
                p5.loadImage(url,
                    // type assertion asserts that the canvas property is present
                    (img) => resolve(img as P5Image),
                    () => reject(new Error("Error creating p5 Image"))
                );
            }
        });
    };

    /**
     * Create a p5.Image from an array of pixels, along with width and height.
     * @param width
     * @param height
     * @param data
     */
    fromPixels({width, height, data}: ImageData): P5Image {
        const p5 = this.p5Instance;
        if (!p5 || !p5.createImage) {
            throw new Error("p5 not loaded");
        }
        /*
         * Before accessing the pixels of an image, the data must loaded with the loadPixels() function.
         * After the array data has been modified, the updatePixels() function must be run to update the changes.
         */
        const img = p5.createImage(width, height);
        img.loadPixels();
        // note: cannot just overwrite the whole pixels array
        data.forEach((value, i) => img.pixels[i] = value);
        img.updatePixels();
        // type assertion asserts that the canvas property is present
        return img as P5Image;
    }

    /**
     * convert raw bytes to blob object
     * @param {Array} raws
     * @param {number} x
     * @param {number} y
     * @returns {Blob}
     */
    async rawToBlob(raws: number[] | Iterable<number> | ArrayLike<number>, x: number, y: number): Promise<Blob> {
        const arr = Array.from(raws)
        const canvas = document.createElement('canvas'); // Consider using offScreenCanvas when it is ready?
        const ctx = canvas.getContext('2d');

        canvas.width = x;
        canvas.height = y;

        const imgData = ctx.createImageData(x, y);
        const {data} = imgData;

        for (let i = 0; i < x * y * 4; i += 1) data[i] = arr[i];
        ctx.putImageData(imgData, 0, 0);

        const blob = await this.getBlob(canvas);
        return blob;
    };

    /**
     *  Conver Blob to P5.Image
     * @param {Blob} blob
     * @param {Object} p5Img
     */
    async blobToP5Image(blob: Blob) {
        if (this.checkP5()) {
            const p5Img = await this.loadAsync(URL.createObjectURL(blob));
            return p5Img;
        }
        return null;
    };

}

const p5Utils = new P5Util();

export default p5Utils;