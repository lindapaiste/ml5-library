import {browser, Tensor3D} from "@tensorflow/tfjs-core";
import {createSizedCanvas, getCtx, ImageArg, MediaElement, TfImageSource, VideoArg} from "./imageUtilities";
import {Tensor} from "@tensorflow/tfjs";
import p5 = require("p5");

/**
 * expect to implement all methods -- throw an error if cannot convert
 */
interface ImageWrapper {
    getWidth(): number;

    getHeight(): number;

    toTensor(): Tensor3D;

    toBlob(type?: string): Blob | Promise<Blob>;

    toData(): ImageData | Promise<ImageData>;

    toPixels(): Uint8ClampedArray | Promise<Uint8ClampedArray>;

    toCanvas(): HTMLCanvasElement | Promise<HTMLCanvasElement>;

    toImage(type?: string): HTMLImageElement | Promise<HTMLImageElement>;

    toDataUrl(type?: string): string | Promise<string>;
}

abstract class ImageWrapperClass<T extends TfImageSource> {

    // Store the Canvas to prevent duplicate creation,
    // Since so many other methods rely on it.
    protected canvas?: HTMLCanvasElement;

    protected internal: T;

    constructor(image: T) {
        this.internal = image;
    }

    abstract getWidth(): number;

    abstract getHeight(): number;

    // Base class methods rely on conversion through Canvas.
    // Can override if there is a better way.

    abstract createCanvas(): HTMLCanvasElement | Promise<HTMLCanvasElement>;

    async toCanvas(): Promise<HTMLCanvasElement> {
        if ( this.canvas ) {
            return this.canvas;
        }
        const canvas = await this.createCanvas();
        this.canvas = canvas;
        return canvas;
    }

    async toBlob(type?: string): Promise<Blob> {
        const canvas = await this.toCanvas();
        // canvas has a toBlob method but it is a sync method that takes a callback
        return new Promise((resolve, reject) => {
            canvas.toBlob(blob => {
                blob === null ? reject(new Error("Error converting canvas to Blob")) : resolve(blob);
            }, type); // implied image/png format
        });
    }

    async toData(): Promise<ImageData> {
        const canvas = await this.toCanvas();
        return getCtx(canvas).getImageData(0, 0, this.getWidth(), this.getHeight());
    }

    async toPixels(): Promise<Uint8ClampedArray> {
        const data = await this.toData();
        return data.data;
    }

    async toDataUrl(type?: string): Promise<string> {
        return (await this.toCanvas()).toDataURL(type);
    }

    async toImage(type?: string): Promise<HTMLImageElement> {
        const dataUrl = await this.toDataUrl(type);
        const image = document.createElement('img');
        image.src = dataUrl;
        image.width = this.getWidth();
        image.height = this.getHeight();
        return image;
    }

}

class TensorWrapper extends ImageWrapperClass<Tensor3D> implements ImageWrapper {
    // use instance vars to prevent duplicate calls of browser.toPixels
    protected pixels?: Uint8ClampedArray;
    protected canvas?: HTMLCanvasElement;

    getWidth(): number {
        return this.internal.shape[1];
    }

    getHeight(): number {
        return this.internal.shape[0];
    }

    toTensor(): Tensor3D {
        return this.internal;
    }

    private async _loadPixels(): Promise<{ canvas: HTMLCanvasElement; pixels: Uint8ClampedArray }> {
        // tf.browser.toPixels will load pixels into the provided canvas, but returns a Unit8ClampedArray
        this.canvas = document.createElement('canvas');
        this.pixels = await browser.toPixels(this.internal, this.canvas);
        return {
            canvas: this.canvas,
            pixels: this.pixels
        };
    }

    async toPixels(): Promise<Uint8ClampedArray> {
        if ( this.pixels) {
            return this.pixels;
        }
        return (await this._loadPixels()).pixels;
    }

    async toData(): Promise<ImageData> {
        const data = await this.toPixels();
        return {
            data,
            width: this.getWidth(),
            height: this.getHeight()
        }
    }

    async createCanvas(): Promise<HTMLCanvasElement> {
        return (await this._loadPixels()).canvas;
    }
}

class ImageDataWrapper extends ImageWrapperClass<ImageData> implements ImageWrapper {
    getWidth(): number {
        return this.internal.width;
    }

    getHeight(): number {
        return this.internal.height;
    }

    toTensor(): Tensor3D {
        return browser.fromPixels(this.internal);
    }

    async toData(): Promise<ImageData> {
        return this.internal;
    }

    createCanvas(): HTMLCanvasElement {
        const canvas = createSizedCanvas(this);
        const ctx = getCtx(canvas);
        ctx.putImageData(this.internal, 0, 0);
        return canvas;
    }
}

class ElementWrapper<T extends MediaElement> extends ImageWrapperClass<T> implements ImageWrapper {

    isVideo(): this is this & ImageWrapperClass<HTMLVideoElement> {
        return this.internal instanceof HTMLVideoElement;
    }

    getWidth(): number {
        return this.isVideo() ? this.internal.videoWidth : this.internal.width;
    }

    getHeight(): number {
        return this.isVideo() ? this.internal.videoHeight : this.internal.height;
    }

    toTensor(): Tensor3D {
        return browser.fromPixels(this.internal);
    }

    createCanvas(): HTMLCanvasElement {
        const canvas = createSizedCanvas(this);
        const ctx = getCtx(canvas);
        ctx.drawImage(this.internal, 0, 0);
        return canvas;
    }

    async toImage(type?: string): Promise<HTMLImageElement> {
        // could throw error on video, but probably it's actually ok to covert it.
        if ( this.internal instanceof HTMLImageElement ) {
            return this.internal;
        }
        return super.toImage();
    }
}

class CanvasWrapper extends ElementWrapper<HTMLCanvasElement> {
    createCanvas(): HTMLCanvasElement {
        return this.internal;
    }
}

class P5ImageWrapper {

}

const hasProperty = <T extends any, P extends string>(image: T, property: P): image is T & Record<P, any> => {
    return typeof image === "object" && image !== null && property in image;
}

export const isImageData = (image: object) => {
}

/**
 * Certain properties, such as the underlying `canvas`, aren't documented because they are considered private
 */
interface P5Image extends p5.Image {
    canvas: HTMLCanvasElement;
    imageData: ImageData;
}

interface P5Element<T> extends p5.Element {
    elt: T;
}

export const isP5Image = (image: object): image is P5Image => {
    return "canvas" in image && "imageData" in image;
}

// TODO: consistency between minimal and complete p5 objects
// does the wrapped version need to return the original?  Or can it just discard p5 and use the underlying element?
export type Convertible = TfImageSource | P5Element<TfImageSource> | P5Image | {elt: TfImageSource} | {canvas: HTMLCanvasElement};

const wrap = (image: Convertible): ImageWrapper => {

    /* TODO: guard instance checks
    tf code:
if ((pixels as PixelData).data instanceof Uint8Array) {
    isPixelData = true;
} else if (
    typeof (ImageData) !== 'undefined' && pixels instanceof ImageData) {
    isImageData = true;
} else if (
    typeof (HTMLVideoElement) !== 'undefined' &&
    pixels instanceof HTMLVideoElement) {
    isVideo = true;
} else if (
    typeof (HTMLImageElement) !== 'undefined' &&
    pixels instanceof HTMLImageElement) {
*/
    if ( "canvas" in image ) {
        return new CanvasWrapper(image.canvas);
    }
    if ( "elt" in image ) {
        return wrap(image.elt);
    }
    if (image instanceof Tensor) {
        return new TensorWrapper(image);
    }
    if (image instanceof HTMLCanvasElement) {
        return new CanvasWrapper(image);
    }
    if (image instanceof HTMLImageElement || image instanceof HTMLVideoElement) {
        return new ElementWrapper(image);
    }
    if (image instanceof ImageData) {
        return new ImageDataWrapper(image);
    }

    else {
        throw new Error("Invalid image type");
        // TODO: better error message
        /*
        throw new Error(
        'pixels passed to tf.browser.fromPixels() must be either an ' +
        `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
        `in browser, or OffscreenCanvas, ImageData in webworker` +
        ` or {data: Uint32Array, width: number, height: number}, ` +
        `but was ${(pixels as {}).constructor.name}`);
         */
    }
}

export default wrap;

export const toBlob = (image: Convertible, type?: string) => {
    return wrap(image).toBlob(type);
}

export const toTensor = (image: Convertible) => {
    return wrap(image).toTensor();
}

export const toImageData = (image: Convertible) => {
    return wrap(image).toData();
}

export const toPixels = async (image: Convertible): Promise<Uint8ClampedArray> => {
    const imageData = await toImageData(image);
    return imageData.data;
}

export const toImage = (image: Convertible, type?: string) => {
    return wrap(image).toImage(type);
}