import callCallback, {Callback} from "./callcallback";
import {EventEmitter} from "events";
import {TfImageSource, videoLoaded} from "./imageUtilities";
import {Convertible} from "./imageConversion";
import {ArgSeparator, BasicArgs} from "./argSeparator";
import * as tf from "@tensorflow/tfjs";

type InnerModel<T> = T extends {model?: infer M} ? M : never;

type Constructor<T> = new (...args: any) => T;

type ModelCallback<M> = Callback<{
    modelReady: boolean;
    ready: Promise<any>; // is sometimes specified in docs as a boolean
    model?: InnerModel<M>;
}> // TODO: what about methods?

/**
 * For compatibility with existing setup.
 * Combine the model and the create function into a class which is compatible with the previous class
 * and can be used interchangeably.
 */
const createModelWrapper = <Model, F extends (...args: any[]) => Model>(creatorFunction: F, modelConstructor: Constructor<Model>) => {
    class ModelWrapper {
        public modelReady: boolean = false;
        public ready: Promise<this>; // is sometimes specified in docs as a boolean
        // for backwards compatibility, if there is a model property it should be to the inner model of the typed model
        public model?: InnerModel<Model>;
        private instance?: Model;

        // takes the arguments of the creator function and adds the callback as an optional final argument
        constructor(...args: [...Parameters<F>, ModelCallback<Model>] | [...Parameters<F>]) {
            const last = args[args.length - 1];
            const callback = typeof last === "function" ? last as ModelCallback<Model> : undefined;
            this.ready = callCallback((async () => {
                this.instance = await creatorFunction(...args);
                this.model = (this.instance as {model?: InnerModel<Model>}).model;
                this.modelReady = true;
                return this;
            })(), callback)
        }
    }

    // TODO: add methods through Object.prototype manipulation
    /*
     * async generate(label: string, callback?: Callback<GeneratedImageResult>) {
            await this.ready;
            return callCallback(this.instance!.generate(label), callback);
        }
     */

    return ModelWrapper;
}

// instead of calling the creator function, just take the result
export class AroundModel<Model> {
    public modelReady: boolean = false;
    public ready: Promise<this>; // is sometimes specified in docs as a boolean
    // for backwards compatibility, if there is a model property it should be to the inner model of the typed model
    public model?: InnerModel<Model>;
    private instance?: Model;
    private modelPromise: Promise<Model>;

    constructor(promise: Promise<Model>) {
        this.modelPromise = promise;
        this.ready = promise.then((result) => {
            this.modelReady = true;
            this.instance = result;
            this.model = (result as {model?: InnerModel<Model>}).model;
            return this;
        });
    }
}

type ConstructorArgs<Options> = BasicArgs & {
    options?: Options;
}

type Wrapped<T> = {
    (): Promise<T>;
    (input: Convertible): Promise<T>;
    (callback: Callback<T>): Promise<T>;
    (input: Convertible, callback: Callback<T>): Promise<T>;
}

export abstract class AbstractImageVideoModel<Model, Options> extends EventEmitter {
    public modelReady: boolean = false;
    public ready: Promise<this>; // is sometimes specified in docs as a boolean
    public model?: Model;
    private readonly modelPromise: Promise<Model>;
    public config: Options;
    public video?: HTMLVideoElement;

    constructor({options, video}: { options?: Options; video?: HTMLVideoElement } | undefined = {}) {
        super();

        this.video = video;

        this.config = {
            ...this.defaultConfig(),
            ...options
        }

        this.modelPromise = this.loadModel();

        this.ready = this.init();
    }

    private async init(): Promise<this> {
        this.model = await this.modelPromise;
        this.modelReady = true;
        return this;
    }

    abstract loadModel(): Promise<Model>;

    abstract defaultConfig(): Options;


    // TODO: which ones actually use options?
    /**
     * innerMethod gets the model as an argument in order to ensure that the model is always defined before it is called.
     */
    protected _makeImageMethod<T>(innerMethod: (model: Model, image: TfImageSource) => Promise<T>, eventName?: string): Wrapped<T> {
        return async (inputOrCallback?: Convertible | Callback<T>, cb?: Callback<T>): Promise<T> => {
            const {image, callback} = new ArgSeparator(this.video, inputOrCallback, cb);
            return callCallback((async () => {
                if (!image) {
                    throw new Error("No image or video found. An image must be provided if no video was set in the constructor.")
                }
                // TODO: figure out video frame handling
                if (image instanceof HTMLVideoElement) {
                    await videoLoaded(image);
                    await tf.nextFrame();
                }
                const model = await this.modelPromise;
                const result = await innerMethod(model, image);
                if (eventName) {
                    this.emit(eventName, result);
                }
                return result;
            })(), callback);
        }
    }

    protected _makeImageMethod2<T>(innerMethod: (this: this & {model: Model}, image: TfImageSource) => Promise<T>, eventName?: string): Wrapped<T> {
        return async (inputOrCallback?: Convertible | Callback<T>, cb?: Callback<T>): Promise<T> => {
            const {image, callback} = new ArgSeparator(this.video, inputOrCallback, cb);
            return callCallback((async () => {
                if (!image) {
                    throw new Error("No image or video found. An image must be provided if no video was set in the constructor.")
                }
                // TODO: figure out video frame handling
                if (image instanceof HTMLVideoElement) {
                    await videoLoaded(image);
                    await tf.nextFrame();
                }
                const model = await this.modelPromise;
                const result = await innerMethod(model, image);
                if (eventName) {
                    this.emit(eventName, result);
                }
                return result;
            })(), callback);
        }
    }
}

export type ValueOf<T> = T[keyof T];

/**
 * Couldn't pass Callback to the constructor due to Typescript this type issues, so call it here.
 */
export const createFactory = <Model, Options, Instance extends AbstractImageVideoModel<Model, Options>>(constructor: new (obj: ConstructorArgs<Options>) => Instance) => {
    // TODO: clean up these types for better inference
    function factory(...args: Array<ValueOf<ConstructorArgs<Options>>>): Promise<Instance>;
    function factory(...args: Array<ValueOf<ConstructorArgs<Options>> | Callback<Instance>>): Instance;
    function factory(...args: any[]) {
        const {callback, ...rest} = new ArgSeparator(...args);

        const instance = new constructor(rest as ConstructorArgs<Options>);
        if ( callback) {
            callCallback(instance.ready, callback);
            return instance;
        }
        else {
            return callCallback(instance.ready);
        }
    }

    return factory;
}

export const