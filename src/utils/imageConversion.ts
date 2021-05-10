import {browser, image, Rank, tensor, Tensor2D, Tensor3D} from "@tensorflow/tfjs-core";
import {createSizedCanvas, getCtx, MediaElement, TfImageSource} from "./imageUtilities";
import {Tensor} from "@tensorflow/tfjs";
import {P5Element, P5Image} from "./p5Utils";

/**
 * Note: uses prefixed names, ie. "imageType" instead of "type" to remove ambiguity
 * so that model options can extend this.
 */
export interface ImageConvertOptions {
    /**
     * The image format to use for the Blob and DataURL.
     * Default: image/png
     */
    imageType?: string;
    /**
     * A Number between 0 and 1 indicating the image quality to use for image formats that use lossy compression
     * such as image/jpeg and image/webp.
     * Default: 0.92
     */
    imageQuality?: number;
}

/**
 * expect to implement all methods -- throw an error if cannot convert
 */
interface ImageWrapper {
    getWidth(): number;

    getHeight(): number;

    toTensor(): Tensor3D;

    toBlob(): Blob | Promise<Blob>;

    toData(): ImageData | Promise<ImageData>;

    toPixels(): Uint8ClampedArray | Promise<Uint8ClampedArray>;

    toCanvas(): HTMLCanvasElement | Promise<HTMLCanvasElement>;

    toImage(): HTMLImageElement | Promise<HTMLImageElement>;

    toDataUrl(): string | Promise<string>;
}

/**
 * Most other conversions go through canvas. Most conversions from canvas are synchronous (except blob).
 */
class CanvasConverter implements ImageWrapper {
    constructor(protected canvas: HTMLCanvasElement, public options: ImageConvertOptions = {}) {
    }

    getWidth(): number {
        return this.canvas.width;
    }

    getHeight(): number {
        return this.canvas.height;
    }

    toTensor(): Tensor3D {
        return browser.fromPixels(this.canvas);
    }

    toBlob(): Promise<Blob> {
        // canvas has a toBlob method but it is a sync method that takes a callback
        return new Promise((resolve, reject) => {
            this.canvas.toBlob(blob => {
                blob === null ? reject(new Error("Error converting canvas to Blob")) : resolve(blob);
            }, this.options.imageType, this.options.imageQuality);
        });
    }

    toData(): ImageData {
        return getCtx(this.canvas).getImageData(0, 0, this.getWidth(), this.getHeight());
    }

    toPixels(): Uint8ClampedArray {
        return this.toData().data;
    }

    toCanvas(): HTMLCanvasElement {
        return this.canvas;
    }

    toImage(): HTMLImageElement {
        const dataUrl = this.toDataUrl();
        const image = document.createElement('img');
        image.src = dataUrl;
        image.width = this.getWidth();
        image.height = this.getHeight();
        return image;
    }

    toDataUrl(): string {
        return this.canvas.toDataURL(this.options.imageType, this.options.imageQuality);
    }
}

type WrapperSource = TfImageSource | Tensor2D;

abstract class ImageWrapperClass<T extends WrapperSource> {

    // Store the Canvas to prevent duplicate creation since so many other methods rely on it.
    protected canvas?: CanvasConverter;

    protected internal: T;

    public options: ImageConvertOptions;

    constructor(image: T, options: ImageConvertOptions = {}) {
        this.internal = image;
        this.options = options;
    }

    setType(type: string): void {
        this.options.imageType = type;
    }

    setQuality(quality: number): void {
        this.options.imageQuality = quality;
    }

    abstract getWidth(): number;

    abstract getHeight(): number;

    abstract createCanvas(): HTMLCanvasElement | Promise<HTMLCanvasElement>;

    // Base class methods rely on conversion through Canvas.
    // Can override if there is a better way.

    private async getCanvas(): Promise<CanvasConverter> {
        if (this.canvas) {
            return this.canvas;
        }
        const canvas = await this.createCanvas();
        this.canvas = new CanvasConverter(canvas, this.options);
        return this.canvas;
    }

    async toCanvas(): Promise<HTMLCanvasElement> {
        return (await this.getCanvas()).toCanvas();
    }

    async toBlob(): Promise<Blob> {
        return (await this.getCanvas()).toBlob();
    }

    async toData(): Promise<ImageData> {
        return (await this.getCanvas()).toData();
    }

    async toPixels(): Promise<Uint8ClampedArray> {
        return (await this.getCanvas()).toPixels();
    }

    async toDataUrl(): Promise<string> {
        return (await this.getCanvas()).toDataUrl();
    }

    async toImage(): Promise<HTMLImageElement> {
        return (await this.getCanvas()).toImage();
    }

}

export class TensorWrapper<T extends Tensor2D | Tensor3D = Tensor3D> extends ImageWrapperClass<T> implements ImageWrapper {
    // use instance vars to prevent duplicate calls of browser.toPixels
    protected pixels?: Uint8ClampedArray;
    protected canvas?: CanvasConverter;

    getWidth(): number {
        return this.internal.shape[1];
    }

    getHeight(): number {
        return this.internal.shape[0];
    }

    toTensor(): Tensor3D {
        // Not sure how to handle a 2D tensor here.
        // Can simply reshape to a 3D with depth 1. Could also convert to a grayscale image and convert that to a tensor.
        const tensor = this.internal;
        return is3D(tensor) ? tensor : tensor.reshape([this.getWidth(), this.getHeight(), 1]);
    }

    private async _loadPixels(): Promise<{ canvas: HTMLCanvasElement; pixels: Uint8ClampedArray }> {
        // tf.browser.toPixels will load pixels into the provided canvas, but returns a Unit8ClampedArray
        const canvas = document.createElement('canvas');
        this.pixels = await browser.toPixels(this.internal, canvas);
        this.canvas = new CanvasConverter(canvas);
        return {
            canvas,
            pixels: this.pixels
        };
    }

    async toPixels(): Promise<Uint8ClampedArray> {
        if (this.pixels) {
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
        if (this.internal instanceof HTMLImageElement) {
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

const hasProperty = <T extends any, P extends string>(image: T, property: P): image is T & Record<P, any> => {
    return typeof image === "object" && image !== null && property in image;
}

export const isImageData = (image: object) => {
}

export const isP5Image = (image: object): image is P5Image => {
    return "canvas" in image && "imageData" in image;
}

export const isRank = <R extends Rank>(tensor: Tensor, rank: R): tensor is Tensor<R> => {
    return tensor.rankType === rank;
}

export const is3D = (tensor: Tensor): tensor is Tensor3D => {
    return tensor.rankType === Rank.R3;
}

export const is2D = (tensor: Tensor): tensor is Tensor2D => {
    return tensor.rankType === Rank.R2;
}




// TODO: consistency between minimal and complete p5 objects
// does the wrapped version need to return the original?  Or can it just discard p5 and use the underlying element?
export type Convertible =
    Tensor
    | TfImageSource
    | P5Element<MediaElement>
    | P5Image
    | { elt: MediaElement }
    | { canvas: HTMLCanvasElement };

/**
 * Overloads allow for the specific type to be known.
 * @param image
 * @param options
 */
function wrap(image: Tensor, options?: ImageConvertOptions): TensorWrapper;
function wrap(image: { elt: HTMLCanvasElement } | { canvas: HTMLCanvasElement } | P5Image, options?: ImageConvertOptions): CanvasWrapper;
function wrap<T extends MediaElement>(image: T | { elt: T } | P5Element<T>, options?: ImageConvertOptions): ElementWrapper<T>;
function wrap(image: ImageData, options?: ImageConvertOptions): ImageDataWrapper;
function wrap(image: Convertible, options?: ImageConvertOptions): ImageWrapper;
function wrap(image: Convertible, options?: ImageConvertOptions): ImageWrapper {

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
    if ("canvas" in image) {
        return new CanvasWrapper(image.canvas, options);
    }
    if ("elt" in image) {
        return wrap(image.elt, options);
    }
    if (image instanceof Tensor) {
        if (!(is3D(image) || is2D(image))) {
            throw new Error(`Invalid tensor. Image tensor must be rank R3 (color) or R2 (grayscale), but encountered rank ${image.rankType}`);
        }
        return new TensorWrapper(image, options);
    }
    if (image instanceof HTMLCanvasElement) {
        return new CanvasWrapper(image, options);
    }
    if (image instanceof HTMLImageElement || image instanceof HTMLVideoElement) {
        return new ElementWrapper(image, options);
    }
    if (image instanceof ImageData) {
        return new ImageDataWrapper(image, options);
    } else {
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

export type SyncConvertible = Exclude<Convertible, Tensor>

// TODO: figure this out better and combine with above
export const toSyncConverter = (image: SyncConvertible, options?: ImageConvertOptions): CanvasConverter => {
    if ("canvas" in image) {
        return new CanvasConverter(image.canvas, options);
    }
    if ("elt" in image) {
        return toSyncConverter(image.elt, options);
    }
    if (image instanceof HTMLCanvasElement) {
        return new CanvasConverter(image, options);
    }
    if (image instanceof HTMLImageElement || image instanceof HTMLVideoElement) {
        const wrapper = new ElementWrapper(image, options);
        return new CanvasConverter(wrapper.createCanvas(), options);
    }
    if (image instanceof ImageData) {
        const wrapper = new ImageDataWrapper(image, options);
        return new CanvasConverter(wrapper.createCanvas(), options);
    } else {
        throw new Error("Invalid image type");
    }
}

export default wrap;

// Quick & dirty helpers.  Will not be performant when doing multiple conversions for the same image.

export const toBlob = (image: Convertible) => {
    return wrap(image).toBlob();
}

export const toTensor = (image: Convertible) => {
    return wrap(image).toTensor();
}

export const toImageData = (image: Convertible) => {
    return wrap(image).toData();
}

export const toPixels = (image: Convertible) => {
    return wrap(image).toPixels();
}

export const toImage = (image: Convertible) => {
    return wrap(image).toImage();
}