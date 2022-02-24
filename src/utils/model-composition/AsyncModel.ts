import callCallback, {Callback} from "../callcallback";
import {EventEmitter} from "events";


export type Ready<T extends { model?: any }> = T & {
    model: NonNullable<T['model']>
}

export abstract class AsyncModel<Model, Options> extends EventEmitter {
    public modelReady: boolean = false;
    public ready: Promise<Ready<this>>; // is sometimes specified in docs as a boolean
    public model?: Model;
    private readonly modelPromise: Promise<Model>;
    public config: Options;

    constructor(options: Partial<Options> = {}) {
        super();

        this.config = {
            ...this.defaultConfig(),
            ...options
        }

        this.modelPromise = this.loadModel();

        this.ready = this.init();
    }

    private async init(): Promise<Ready<this>> {
        this.model = await this.modelPromise;
        this.modelReady = true;
        this.emit("ready", this);
        return this as Ready<this>;
    }

    abstract loadModel(): Promise<Model>;

    abstract defaultConfig(): Options;

    protected callWhenReady<A extends any[], T>(innerMethod: (this: Ready<this>, ...args: A) => Promise<T>, callback: Callback<T> | undefined, eventName: string, ...args: A) {
        return callCallback((async () => {
            const ready = await this.ready;
            const result = await innerMethod.call(ready, ...args);
            this.emit(eventName, result);
            return result;
        })(), callback);
    }
}


/**
 * Needs to return the instance when there is a callback,
 * but return a Promise if there is no callback.
 *
 * TODO: better factory
 */
export const constructWithCallback = <Model, Options, Instance extends AsyncModel<Model, Options>>(
    constructor: new (options?: Options) => Instance,
    options?: Options,
    callback?: Callback<Ready<Instance>>
) => {
    const instance = new constructor(options);
    if (callback) {
        callCallback(instance.ready, callback);
        return instance;
    } else {
        return callCallback(instance.ready);
    }
}