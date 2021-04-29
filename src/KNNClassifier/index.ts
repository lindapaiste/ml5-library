// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
 * A K-nearest neighbors (KNN) classifier that allows fast
 * custom model training on top of any tensor input. Useful for transfer
 * learning with an embedding from another pretrained model.
*/

import * as tf from '@tensorflow/tfjs';
import {Rank, Tensor, TensorLike} from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as io from '../utils/io';
import callCallback, {Callback} from '../utils/callcallback';
import {Tensor2D} from "@tensorflow/tfjs-core";
import {ArgSeparator} from "../utils/argSeparator";

interface KNNClassification {
    label: string;
    classIndex: number;
    confidences: Record<string, number>;
    confidencesByLabel: Record<string, number>;
}

interface KNNSavedData {
    tensors: (Float32Array | Int32Array | Uint8Array)[];
    dataset: Record<string,
        // serialized Tensor + added label
        Pick<Tensor2D, 'shape' | 'dtype'> & {label: string}
    >
}

export const asTensor = (input: Tensor | TensorLike): Tensor => {
    return input instanceof Tensor ? input : tf.tensor(input);
}

// TODO: it seems like the internal tersorflow model already supports string labels.  Is mapStringToIndex even necessary?

/**
 * @property {KNNClassifier} knnClassifier
 */
class KNN {
    // TODO: rename to `model` for consistency
    public readonly knnClassifier: knnClassifier.KNNClassifier;
    private mapStringToIndex: string[];


    /**
     * Create a KNNClassifier instance.
     */
    constructor() {
        this.knnClassifier = knnClassifier.create();
        this.mapStringToIndex = [];
    }

    /**
     * Adding an example to a class.
     * @param {*} input - An example to add to the dataset, usually an activation from another model.
     * @param {(number | string)} classIndexOrLabel  The class index(number) or label(string) of the example.
     */
    addExample(input: Tensor | TensorLike, classIndexOrLabel: number | string) {
        let classIndex = this.labelIndex(classIndexOrLabel);

        // add to labels map if not already present
        if (typeof classIndexOrLabel === 'string' && classIndex === -1) {
            classIndex = this.mapStringToIndex.push(classIndexOrLabel) - 1;
        }

        // convert to Tensor and add to TensorFlow model
        this.knnClassifier.addExample(asTensor(input), classIndex);
    }

    /**
     * Classify an new input. It returns an object with a top classIndex and label, confidences mapping all class indices to their confidence, and confidencesByLabel mapping all classes' confidence by label.
     * @param {*} input  - An example to make a prediction on, could be an activation from another model or an array of numbers.
     * @param {number} kOrCallback  - Optional. The K value to use in K-nearest neighbors. The algorithm will first find the K nearest examples from those it was previously shown, and then choose the class that appears the most as the final prediction for the input example. Defaults to 3. If examples < k, k = examples.
     * @param {function} cb  - Optional. A function to be called once the input has been classified. If no callback is provided, it will return a promise that will be resolved once the model has classified the new input.
     */
    async classify(input: Tensor | TensorLike, kOrCallback: number | Callback<KNNClassification>, cb?: Callback<KNNClassification>) {
        const {number: k = 3, callback} = new ArgSeparator(kOrCallback, cb);
        return callCallback(this.classifyInternal(asTensor(input), k), callback);
    }

    /**
     * @param {*} input
     * @param {number} k
     * @return {Promise<{label: string, classIndex: number, confidences: {[p: string]: number}}>}
     */
    async classifyInternal(input: Tensor, k: number): Promise<KNNClassification> {
        const numClass = this.knnClassifier.getNumClasses();
        if (numClass <= 0) {
            throw new Error('There is no example in any class');
        } else {
            const res = await this.knnClassifier.predictClass(input, k);
            if (this.mapStringToIndex.length > 0) {
                if (res.classIndex || res.classIndex === 0) {
                    const label = this.mapStringToIndex[res.classIndex];
                    if (label) res.label = label;
                }
                if (res.confidences) {
                    res.confidencesByLabel = {};
                    const {confidences} = res;
                    const indexes = Object.keys(confidences);
                    indexes.forEach((index) => {
                        const label = this.mapStringToIndex[index];
                        res.confidencesByLabel[label] = confidences[index];
                    });
                }
            }
            return res;
        }
    }

    /**
     * Helper method converts an argument which is either a label or an index into an index
     * @param labelOrIndex
     * @private
     */
    private labelIndex(labelOrIndex: string | number): number {
        return typeof labelOrIndex === "number" ? labelOrIndex : this.mapStringToIndex.indexOf(labelOrIndex);
    }

    /**
     * Clear all examples in a label.
     * @param {number||string} labelOrIndex - The class index or label, a number or a string.
     */
    clearLabel(labelOrIndex: string | number) {
        const classIndex = this.labelIndex(labelOrIndex);
        // throw error if -1?  do nothing?
        this.knnClassifier.clearClass(classIndex);
    }

    clearAllLabels(): void {
        this.mapStringToIndex = [];
        this.knnClassifier.clearAllClasses();
    }

    /**
     * Get the example count for each label. It returns an object that maps class label to example count for each class.
     */
    getCountByLabel(): Record<string, number> {
        const countByIndex = this.knnClassifier.getClassExampleCount();
        if (this.mapStringToIndex.length > 0) {
            const countByLabel: Record<string, number> = {};
            Object.keys(countByIndex).forEach((key) => {
                // @ts-ignore the keys are of type string, but they should be numbers like "1"
                if (this.mapStringToIndex[key]) {
                    // @ts-ignore
                    const label = this.mapStringToIndex[key];
                    countByLabel[label] = countByIndex[key];
                }
            });
            return countByLabel;
        }
        return countByIndex;
    }

    /**
     * Get the example count for each class. It returns an object that maps class index to example count for each class.
     */
    getCount(): Record<string, number> {
        return this.knnClassifier.getClassExampleCount();
    }

    getClassifierDataset(): Record<string, Tensor2D> {
        return this.knnClassifier.getClassifierDataset();
    }

    setClassifierDataset(dataset: Record<string, Tensor2D>): void {
        this.knnClassifier.setClassifierDataset(dataset);
    }

    /**
     * It returns the total number of labels.
     */
    getNumLabels(): number {
        return this.knnClassifier.getNumClasses();
    }

    dispose(): void {
        this.knnClassifier.dispose();
    }

    /**
     * Download the whole dataset as a JSON file. It's useful for saving state.
     * @param {string} [name] - Optional. The name of the JSON file that will be downloaded. e.g. "myKNN" or "myKNN.json". If no fileName is provided, the default file name is "myKNN.json".
     */
    async save(name?: string): Promise<void> {
        const dataset = this.knnClassifier.getClassifierDataset();
        if (this.mapStringToIndex.length > 0) {
            Object.keys(dataset).forEach((key) => {
                if (this.mapStringToIndex[key]) {
                    dataset[key].label = this.mapStringToIndex[key];
                }
            });
        }
        const tensors = Object.keys(dataset).map((key) => {
            const t = dataset[key];
            if (t) {
                return t.dataSync();
            }
            return null;
        });
        let fileName = 'myKNN.json';
        if (name) {
            fileName = name.endsWith('.json') ? name : `${name}.json`;
        }
        await io.saveBlob(JSON.stringify({dataset, tensors}), fileName, 'application/octet-stream');
    }

    /**
     * Load a dataset from a JSON file. It's useful for restoring state.
     * @param {string} pathOrData - The path for a valid JSON file.
     * @param {function} [callback] - Optional. A function to run once the dataset has been loaded. If no callback is provided, it will return a promise that will be resolved once the dataset has loaded.
     */
    async load(pathOrData: string | KNNSavedData, callback: Callback<void>): Promise<void> {
        return callCallback((async () => {
            let data;
            if (typeof pathOrData === 'object') {
                data = pathOrData;
            } else {
                data = await io.loadFile<KNNSavedData>(pathOrData);
            }
                const {dataset, tensors} = data;
                this.mapStringToIndex = Object.keys(dataset).map(key => dataset[key].label);
                const tensorsData = tensors
                    .map((values, i) => tf.tensor<Rank.R2>(values, dataset[i].shape, dataset[i].dtype))
                    .reduce((acc: Record<number, Tensor2D>, cur, j) => {
                        acc[j] = cur;
                        return acc;
                    }, {});
                this.knnClassifier.setClassifierDataset(tensorsData);
        })(), callback);
    }
}

const KNNClassifier = () => new KNN();

export default KNNClassifier;
