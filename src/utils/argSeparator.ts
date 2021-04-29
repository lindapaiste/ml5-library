import {extractImageElement, ImageArg, TfImageSource, VideoArg} from "./imageUtilities";

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

type Element<T> = T extends Array<infer U> ? U : never;

/**
 * Type to extend with args for specific options and callback types
 */
export interface BasicArgs {
    image?: TfImageSource;
    video?: HTMLVideoElement;
    options?: object;
    callback?: Function;
    string?: string;
    number?: number;
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
export class ArgSeparator<Args extends any[]> {

    /**
     * Strict types here mean no errors when destructuring,
     * but makes assignment inside this class tougher.
     */
    image?: TfImageSource;
    video?: HTMLVideoElement;
    options?: Exclude<Extract<Element<Args>, object>, ImageArg | Function>;
    callback?: Extract<Element<Args>, Function>;
    string?: string;
    number?: number;

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
    constructor(...args: Args) {
        args.forEach(this.addArg);
    }

    /**
     * Static constructor for easier chaining.
     */
    static from<T extends any[]>(...args: T): ArgSeparator<T> {
        return new ArgSeparator(...args);
    }

    /**
     * Can add arguments through the constructor or at any time after construction.
     *
     * @param {(ImageArg | VideoArg | Object | function)} arg - a video, callback, or options object
     * @param {number} [index] - optional number used in error messages
     */
    addArg(arg: Element<Args>, index?: number): void {
        // skip over falsey arguments and don't throw any error, assuming that these are omissions
        // do this check first to prevent accessing properties on null, which is an object
        if (arg === undefined || arg === null) {
            return;
        }
        switch (typeof arg) {
            case "string":
                this.string = arg;
                return;
            case "number":
                this.number = arg;
                return;
            case "function":
                this.callback = arg
                return;
            case "object":
                // Videos are also images, but images are not all videos,
                // so keep separate properties but store videos in both
                const image = extractImageElement(arg);
                if (image) {
                    this.image = image;
                    if (image instanceof HTMLVideoElement) {
                        this.video = image;
                    }
                }
                // objects which are not images are assumed to be options
                else {
                    this.options = arg;
                }
                return;
            default:
                // Notify user about invalid arguments (would be ok to just skip)
                throw new InvalidVideoArgError(arg, index);
        }
    }

    /**
     * Check whether or not a given property has been set
     * @param property
     */
    hasProperty<K extends keyof this>(property: K): this is this & Record<K, NonNullable<this[K]>> {
        return this[property] !== undefined;
    }

    /**
     * Check that an argument exists and throw an error if it doesn't
     * @param property
     * @param message
     */
    require<K extends keyof this>(property: K, message?: string): this & Record<K, NonNullable<this[K]>> {
        if ( this.hasProperty(property)) {
            return this;
        }
        throw new Error( message || `An argument for ${property} must be provided.`);
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