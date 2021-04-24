import {Callback} from "./callcallback";
import {extractImageElement, ImageArg, TfImageSource, VideoArg} from "./imageUtilities";

export type VideoModelArg<Instance, Options extends object = {}> = VideoArg | Callback<Instance> | Options;

// TODO: generalize to support method calls
export class InvalidVideoArgError extends TypeError {
    public readonly arg: any;

    constructor(arg: any, i?: number) {
        const message = `Invalid argument passed to model constructor${i === undefined ? '.' : `in position ${i} (zero-indexed).`}.
      Received value: ${String(arg)}.
      Argument must be one of the following types: an HTML video element, a p5 video element, an options object, a callback function.`;
        super(message);
        this.name = 'InvalidVideoArgumentError';
        this.arg = arg;
    }
}

/**
 * Helper utility to parse an array of arguments into known properties
 *
 * All properties are optional as an arguments might be missing
 * @property {TfImageSource} [image]
 * @property {HTMLVideoElement} [video]
 * @property {Object} [options]
 * @property {function} [callback]
 */
export class ImageModelArgs<Callback extends Function, Options extends object = {}> {

    image?: TfImageSource;
    video?: HTMLVideoElement;
    options?: Options;
    callback?: Callback;

    /**
     * Arguments used to CREATE an image-based model can be:
     *  - video: an HTMLVideoElement or p5 video.
     *  - options: an object of options specific to this model.
     *  - callback: a function to run once the model has been loaded. Called with arguments (error) or (error, result).
     *
     * Arguments used to CALL a method an image-based model can be:
     *  - image: an image or video element or an ImageData object.  Valid types: HTMLImageElement, HTMLCanvasElement,
     *    HTMLVideoElement, ImageData, p5 image, p5 video.
     *  - options: an object of options specific to this model.
     *  - callback: a function to run once the method has been completed.
     *
     * Expected to be provided in order video/image, options, callback with any omitted.
     * This function does not actually require any particular order.
     *
     *  @param {(ImageArg | VideoArg | Object | function)[]} args
     */
    constructor(...args: Array<Callback | Options | ImageArg | VideoArg>) {
        args.forEach(this.addArg);
    }

    /**
     * Can add arguments through the constructor or at any time after construction.
     *
     * @param {(ImageArg | VideoArg | Object | function)} arg - a video, callback, or options object
     * @param {number} [index] - optional number used in error messages
     */
    addArg(arg: Callback | Options | ImageArg | VideoArg, index?: number): void {
        // skip over falsey arguments and don't throw any error, assuming that these are omissions
        // do this check first to prevent accessing properties on null, which is an object
        if ( ! arg ) {
            return;
        }
        if (typeof arg === "function") {
            this.callback = arg;
        } else if (typeof arg === "object") {
            // Videos are also images, but images are not all videos,
            // so keep separate properties but store videos in both
            const image = extractImageElement(arg);
            if (image) {
                this.image = image;
                if ( image instanceof HTMLVideoElement ) {
                    this.video = image;
                }
            }
            // objects which are not images are assumed to be options
            else {
                this.options = arg;
            }
        } else {
            // Notify user about invalid arguments (would be ok to just skip)
                throw new InvalidVideoArgError(arg, index);
        }
    }
}


/**
 * Helper utility to parse an array of arguments into known properties
 */
export class ImageMethodArgs<Callback extends Function, Options extends object = {}> {

    image?: TfImageSource;
    options?: Options;
    callback?: Callback;

    /**
     * Arguments used to call a method an image-based model can be:
     *  - image: an image or video element or an ImageData object.  Valid types: HTMLImageElement, HTMLCanvasElement,
     *    HTMLVideoElement, ImageData, p5 image, p5 video.
     *  - options: an object of options specific to this model.
     *  - callback: a function to run once the method has been completed.
     *
     * Expected to be provided in order image, options, callback with any omitted.
     * This function does not actually require any particular order.
     *
     * @param {(ImageArg| Object | function)[]} args
     */
    constructor(...args: Array<Callback | Options | ImageArg>) {
        args.forEach(this.addArg);
    }

    /**
     * Can add arguments through the constructor or at any time after construction.
     *
     * @param {ImageArg| Object | function} arg - an image, callback, or options object
     * @param {number} [index] - optional number used in error messages
     */
    addArg(arg: Callback | Options | ImageArg, index?: number): void {
        if (typeof arg === "function") {
            this.callback = arg;
        } else if (typeof arg === "object") {
            const image = extractImageElement(arg);
            if (image) {
                this.image = image;
            } else {
                this.options = arg;
            }
        } else {
            // Notify user about invalid arguments (would be ok to just skip)
            // But skip over falsey arguments assuming that these are omissions
            if (arg) {
                throw new InvalidVideoArgError(arg, index);
            }
        }
    }
}

/**
 * Can validate that an image or video is among a specific list of supported types.
 * Types must be classes that can be verified with instanceof.
 */
export const createImageChecker = (...validClasses) => ( variable: any ): void => {
    if ( ! validClasses.some( c => variable instanceof c )) {
        throw new Error(`Invalid media type. Must be one of: ${validClasses.map(c => c.name).join(", ")}`);
    }
}