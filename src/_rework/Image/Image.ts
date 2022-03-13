import createHandposeModel from '../../Handpose/rework';
import { getImageElement, handlePolyfill, ImageArg, InputImage } from '../../utils/handleArguments';
import { ImageModel, OptionsFrom, ReturnFrom, Task, Unpromise } from './types';
import { EventEmitter } from 'events';

const ImageModels = {
    detect: {
        hand: createHandposeModel
    }
}

export class ImageModelRunner<T = any, O = any> extends EventEmitter {
    private model: null | ImageModel<T, O> = null;
    private options: Partial<O> = {};
    private media: InputImage | null = null;

    constructor(model: null | ImageModel<T, O> = null) {
        super();
        this.model = model;
    }

    public useModel = <NT, NO>(model: ImageModel<NT, NO>): asserts this is ImageModelRunner<NT, NO> => {
        this.model = model as any;
    }

    public useModel_ = <NT, NO>(model: ImageModel<NT, NO>): ImageModelRunner<NT, NO> => {
        // Can get better TS support when chaining.
        // But also want to allow use without chaining.
        const next = new ImageModelRunner(model);
        next.media = this.media;
        // TODO: what about options type?
        next.options = this.options as any;
        return next;
    }

    private executeTask = async (task: Task): Promise<T> => {
        if (!this.media) {
            throw new Error(`No image found. You must call getMedia() before calling ${task}().`);
        }
        if (!this.model) {
            throw new Error(`No model found. You must call loadModel() or useModel() before calling ${task}().`);
        }
        const fn = this.model?.[task];
        if (!fn) {
            throw new Error(`Current model ${this.model.name} does not support task '${task}'.`);
        }
        const result = await fn(this.media, { ...this.model.defaultOptions, ...this.options });
        this.emit(this.model.event, result);
        return result;
    }

    public classify = () => this.executeTask('classify')

    public generate = () => this.executeTask('generate')

    public stylize = () => this.executeTask('stylize')

    public detect = async <DT extends keyof typeof ImageModels.detect>(detectionType: DT) => {
        if (!this.model) {
            // TODO: need to change type here.
            const model = await ImageModels.detect[detectionType](this.options);
            type Model = Unpromise<ReturnType<typeof ImageModels.detect[DT]>>
            // @ts-ignore
            this.useModel<ReturnFrom<Model>, OptionsFrom<Model>>(model);
            console.log(`No model set. Using default model ${this.model!.name} for detection type ${detectionType}`);
        }
        return this.executeTask('detect');
    }

    public segment = async (segmentationType: string) => {
        // TODO
    }

    // TODO: load from URL, support Tensor
    public getMedia = (media: ImageArg): this => {
        const img = getImageElement(media) || handlePolyfill(media);
        if (!img) {
            throw new Error('Invalid media'); // TODO: list all valid types
        }
        this.media = img;
        return this;
    }
}

const Image = <T, O>() => {
    let model: null | ImageModel<T, O> = null;
    let options: Partial<O> = {};
    let media: InputImage | null = null;

    const executeTask = (task: Task): Promise<T> => {
        if (!media) {
            throw new Error(`No image found. You must call getMedia() before calling ${task}().`);
        }
        if (!model) {
            throw new Error(`No model found. You must call loadModel() or useModel() before calling ${task}().`);
        }
        const fn = model?.[task];
        if (!fn) {
            throw new Error(`Current model ${model.name} does not support task '${task}'.`);
        }
        return fn(media, { ...model.defaultOptions, ...options });
    }

    return {
        classify: executeTask('classify'),
        generate: executeTask('generate'),
        stylize: executeTask('stylize'),
        detect: async (detectionType: keyof typeof ImageModels.detect) => {
            if (!model) {
                // TODO: need to change type here.
                model = await ImageModels.detect[detectionType](options) as any;
                console.log(`No model set. Using default model ${model!.name} for detection type ${detectionType}`);
            }
        },
        segment: async (segmentationType: string) => {
            // TODO
        },
        useModel: <NT, NO>(newModel: ImageModel<NT, NO>) => {
            // TODO: need to change type here.
            model = newModel as any;
        }
    }
}
