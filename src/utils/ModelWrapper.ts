import callCallback, {Callback} from "./callcallback";

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