// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Word2Vec
*/

import * as tf from '@tensorflow/tfjs';
import callCallback, {ML5Callback} from '../utils/callcallback';
import {loadFile} from "../utils/io";
import {ArgSeparator} from "../utils/argSeparator";

interface WordDistance {
    word: string;
    distance: number;
}

/**
 * Instead of using static methods which take a model as an argument, make a class for the model
 */
class Word2VecModel {
    constructor(public readonly words: Record<string, tf.Tensor1D>) {
    }

    add(values: string[]): tf.Tensor1D {
        return this.addOrSubtract(values, 'ADD')
    }

    subtract(values: string[]): tf.Tensor1D {
        return this.addOrSubtract(values, 'SUBTRACT')
    }

    addOrSubtract(values: string[], operation: 'ADD' | 'SUBTRACT'): tf.Tensor1D {
        return tf.tidy(() => {
            if (values.length < 2) {
                throw new Error('Invalid input, must be passed more than 1 value');
            }
            const vectors = values.map(value => {
                if (!(value in this.words)) {
                    throw new Error(`Invalid input, vector not found for: ${value}`);
                }
                return this.words[value];
            })
            const method = operation === 'ADD' ? tf.add : tf.sub;
            return vectors.reduce(method);
        });
    }

    nearest(input: tf.Tensor, start: number, end: number): WordDistance[] {
        return Object.entries(this.words)
            .map(([word, tensor]) => {
                const distance = tf.util.distSquared(input.dataSync(), tensor.dataSync());
                return {word, distance};
            })
            .sort((a, b) => a.distance - b.distance)
            .slice(start, end);
    }
}

class Word2Vec {
    // TODO: want to move fully to separating model from wrapper class
    model: Record<string, tf.Tensor1D>;
    word2vec: Word2VecModel;
    // TODO: can modelSize be removed? It is not documented or used anywhere.
    modelSize: number;
    modelLoaded: boolean;
    ready: Promise<Word2Vec>;

    /**
     * Create Word2Vec model
     * @param {String} modelPath - path to pre-trained word vector model in .json e.g data/wordvecs1000.json
     * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise
     *    that will be resolved once the model has loaded.
     */
    constructor(modelPath: string, callback: ML5Callback<Word2Vec>) {
        this.model = {};
        this.word2vec = new Word2VecModel({});
        this.modelSize = 0;
        this.modelLoaded = false;

        this.ready = callCallback(this.loadModel(modelPath), callback);
        // TODO: Add support to Promise
        // this.then = this.ready.then.bind(this.ready);
    }

    /**
     * Create a 1D Tensor for each word in the model data.  This.model is a dictionary of these Tensors.
     * @param modelPath
     */
    async loadModel(modelPath: string): Promise<this> {
        const data = await loadFile(modelPath);

        Object.keys(data.vectors).forEach((word) => {
            this.model[word] = tf.tensor1d(data.vectors[word]);
        });
        this.modelSize = Object.keys(this.model).length;
        this.word2vec = new Word2VecModel(this.model);
        this.modelLoaded = true;
        return this;
    }

    /**
     * Dispose of all tensors in this.model. Can optionally call a callback.
     * TODO: this isn't async, so what's the point of a callback?
     * @param callback
     */
    dispose(callback?: () => void) {
        Object.values(this.model).forEach(x => x.dispose());
        callback?.();
    }

    async add(inputs: string[], maxOrCb?: number | ML5Callback<WordDistance[]>, cb?: ML5Callback<WordDistance[]>): Promise<WordDistance[]> {
        const {number = 1, callback} = new ArgSeparator(maxOrCb, cb);

        return callCallback((async () => {
            await this.ready;
            //return tf.tidy(() => {
            const sum = this.word2vec.addOrSubtract(inputs, 'ADD');
            return this.word2vec.nearest(sum, inputs.length, inputs.length + number);
            //});
        })(), callback);
    }

    // TODO: can combine with add
    async subtract(inputs: string[], maxOrCb?: number | ML5Callback<WordDistance[]>, cb?: ML5Callback<WordDistance[]>): Promise<WordDistance[]> {
        const {number = 1, callback} = new ArgSeparator(maxOrCb, cb);

        return callCallback((async () => {
            await this.ready;
            //return tf.tidy(() => {
            const subtraction = this.word2vec.addOrSubtract(inputs, 'SUBTRACT');
            return this.word2vec.nearest(subtraction, inputs.length, inputs.length + number);
            //});
        })(), callback);
    }

    async average(inputs: string[], maxOrCb?: number | ML5Callback<WordDistance>[], cb?: ML5Callback<WordDistance>[]): Promise<WordDistance[]> {
        const {number = 1, callback} = new ArgSeparator(maxOrCb, cb);

        return callCallback((async () => {
            await this.ready;
            //return tf.tidy(() => {
            const sum = this.word2vec.addOrSubtract(inputs, 'ADD');
            const avg = tf.div(sum, tf.tensor(inputs.length));
            return this.word2vec.nearest(avg, inputs.length, inputs.length + number);
            //});
        })(), callback);
    }

    async nearest(input: string, maxOrCb?: number | ML5Callback<WordDistance>[], cb?: ML5Callback<WordDistance>[]): Promise<WordDistance[]> {
        const {number = 10, callback} = new ArgSeparator(maxOrCb, cb);

        return callCallback((async () => {
            await this.ready;
            const vector = this.model[input];
            if (!vector) {
                throw new Error(`Input word ${input} not found in model`);
            }
            return this.word2vec.nearest(vector, 1, number + 1);
        })(), callback);
    }

    /* Given a set of your own words, find the nearest neighbors */
    async nearestFromSet(input: string, set: string[], maxOrCb: number | ML5Callback<WordDistance[]>, cb?: ML5Callback<WordDistance[]>): Promise<WordDistance[]> {
        const {number = 10, callback} = new ArgSeparator(maxOrCb, cb);

        return callCallback((async () => {
            await this.ready;
            const vector = this.model[input];

            // If the input vector isn't found, bail out early.
            if (!vector) {
                throw new Error(`Input word ${input} not found in model`);
            }

            // Build a subset of the current model from the provided set.
            const miniModel: Record<string, tf.Tensor1D> = {};
            set.forEach((word) => {
                if (this.model[word]) miniModel[word] = this.model[word];
            });

            // If none of the words in the set are found, also bail out
            // TODO: should they all be required?
            if (!Object.keys(miniModel).length) {
                throw new Error(`No words from set [${set.join(', ')}] could be found on the model.`);
            }

            const instance = new Word2VecModel(miniModel);
            return instance.nearest(vector, 1, number + 1);
        })(), callback);
    }

    async getRandomWord(callback: ML5Callback<string>): Promise<string> {
        return callCallback((async () => {
            await this.ready;
            const words = Object.keys(this.model);
            return words[Math.floor(Math.random() * words.length)];
        })(), callback);
    }

    createCallbackMethod = <M extends (...args: any[]) => any>(innerMethod: M) => {
        // TODO: promisify return type
        return async (...args: Parameters<M> | [...Parameters<M>, ML5Callback<ReturnType<M>>]) => {
            const last = args[args.length - 1];
            const callback = typeof last === "function" ? last as ModelCallback<Model> : undefined;
            return callCallback((async () => {
                await this.ready;
                innerMethod(...args);
            })(), callback);
        }
    }
}

const word2vec = (model: string, cb: ML5Callback<Word2Vec>) => new Word2Vec(model, cb);

export default word2vec;
