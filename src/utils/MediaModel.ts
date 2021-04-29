import {EventEmitter} from "events";
import callCallback, {Callback} from "./callcallback";
import {TfImageSource, VideoArg} from "./imageUtilities";
import MediaWrapper from "./MediaWrapper";
import {Convertible} from "./imageConversion";
import {ArgSeparator} from "./argSeparator";
import * as tf from "@tensorflow/tfjs";

type InnerModel<T> = T extends { model?: infer M } ? M : never;

type Constructor<T> = new (...args: any) => T;

export type Promisify<T> = T extends Promise<any> ? T : Promise<T>;

export type Unpromise<T> = T extends Promise<infer U> ? U : T;

export const createClass = <Model, Options>(creatorFunction: (options: Options) => Promise<Model> | Model, defaultOptions: Options) => {
    return class MediaModel extends EventEmitter {
        public config: Options;
        public modelReady: boolean = false;
        public ready: Promise<this>; // is sometimes specified in docs as a boolean
        // for backwards compatibility, if there is a model property it should be to the inner model of the typed model
        public model?: InnerModel<Model>;
        protected instance?: Model;
        public video?: MediaWrapper;

        // all media models take the same arguments
        constructor(video?: VideoArg, options: Partial<Options> = {}, callback?: Callback<any>) {
            super();
            const merged = {
                ...defaultOptions,
                ...options
            }
            this.config = merged;
            this.ready = callCallback((async () => {
                this.instance = await creatorFunction(merged);
                this.model = (this.instance as { model?: InnerModel<Model> }).model;
                this.modelReady = true;
                return this;
            })(), callback)
        }

        // TODO: which ones actually use options?
        protected _makeImageMethod = <T>(innerMethod: (image: TfImageSource, options?: Partial<Options>) => Promise<T>, eventName?: string) => {
            return async (inputOrCallback?: Convertible | Callback<T>, cb?: Callback<T>): Promise<T> => {
                const {image, callback} = new ArgSeparator(this.video, inputOrCallback, cb);
                if (!image) {
                    throw new Error("No image or video found. An image must be provided if no video was set in the constructor.")
                }
                return callCallback((async () => {
                    // TODO: figure out video frame handling
                    if (image instanceof HTMLVideoElement) {
                        await this.video?.load();
                        await tf.nextFrame();
                    }
                    await this.ready;
                    const result = await innerMethod(image, this.config);
                    if (eventName) {
                        this.emit(eventName, result);
                    }
                    return result;
                })(), callback);
            }
        }
    }
}

type Unpack<T> = T extends Array<infer U> ? U : T;

// can't make create be a static method of the MediaModel because I need the return type to include properties added by extends
export const createFactory = <Args extends any[], C extends new (...args: Args) => { ready: any }>(constructor: C) => {
    function factory(...args: Array<Exclude<Unpack<Args>, Callback<any>>>): InstanceType<C>['ready'];
    function factory(...args: Array<Unpack<Args>>): InstanceType<C>;
    function factory(...args: any[]) {
        const {options, video, callback} = new ArgSeparator(...args);
        // @ts-ignore
        const instance = new constructor(video, options, callback);
        return callback ? instance : instance.ready;
    }

    return factory;
}

/*export const createFactory = <Options, T extends {ready: any}>(constructor: new (video?: VideoArg, options?: Partial<Options>, callback?: Callback<T>) => T ) => {
    function factory(videoOrOptions?: VideoArg | Options , options?: Options): T['ready'];
    function factory(videoOrOptionsOrCallback: VideoArg | Options | Callback<T>, optionsOrCallback?: Options | Callback<T>, cb?: Callback<T>): T;
    function factory(videoOrOptionsOrCallback?: VideoArg | Options | Callback<T>, optionsOrCallback?: Options | Callback<T>, cb?: Callback<T>) {
        const {options, video, callback} = new ArgSeparator(videoOrOptionsOrCallback, optionsOrCallback, cb);
        // @ts-ignore
        const instance = new constructor(video, options, callback);
        return callback ? instance : instance.ready;
    }
    return factory;
}*/

type Instance = EventEmitter & {
    video?: MediaWrapper | HTMLVideoElement | null;
    config: object;
    ready: Promise<any>;
}

export function createImageMethod<T>(this: Instance, innerMethod: (image: TfImageSource, options?: Partial<Instance['config']>) => Promise<T>, eventName?: string) {
    return async (inputOrCallback?: Convertible | Callback<T>, cb?: Callback<T>): Promise<T> => {
        const {image, callback} = new ArgSeparator(this.video, inputOrCallback, cb);
        if (!image) {
            throw new Error("No image or video found. An image must be provided if no video was set in the constructor.")
        }
        return callCallback((async () => {
            // TODO: figure out video frame handling - might be different for passed video vs constructor video
            if (image instanceof HTMLVideoElement) {
                await this.video?.load();
                await tf.nextFrame();
            }
            await this.ready;
            const result = await innerMethod(image, this.config);
            if (eventName) {
                this.emit(eventName, result);
            }
            return result;
        })(), callback);
    }
}