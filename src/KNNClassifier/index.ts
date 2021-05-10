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
import {ObjectBuilder} from "../utils/objectUtilities";

interface KNNClassification {
    /**
     * label/class name for the top classification
     */
    label: string;
    /**
     * index of the top class
     */
    classIndex: number;
    /**
     * confidences for the top K classes, keyed by classIndex
     */
    confidences: Record<number, number>;
    /**
     * confidences for the top K classes, keyed by label
     */
    confidencesByLabel: Record<string, number>;
}

interface SavedDataEntry {
    label: string;
    shape: [number, number];
    dtype: tf.DataType;
    // actually has a few more properties, but they aren't needed
}

interface KNNSavedData {
    /**
     * An array of tensor values which shares the same indexes as the dataset.
     */
    tensors: (Float32Array | Int32Array | Uint8Array)[];
    /**
     * Currently saves as an array, but need to support a numeric-keyed object for backwards compatibility.
     */
    dataset: SavedDataEntry[] | Record<number, SavedDataEntry>;
}

export const asTensor = (input: Tensor | TensorLike): Tensor => {
    return input instanceof Tensor ? input : tf.tensor(input);
}

/**
 * @property {KNNClassifier} knnClassifier
 */
class KNN {
    // TODO: rename to `model` for consistency
    public readonly knnClassifier: knnClassifier.KNNClassifier;


    /**
     * Create a KNNClassifier instance.
     */
    constructor() {
        this.knnClassifier = knnClassifier.create();
    }

    /**
     * Adding an example to a class.
     * @param {*} input - An example to add to the dataset, usually an activation from another model.
     * @param {(number | string)} classIndexOrLabel  The class index(number) or label(string) of the example.
     */
    addExample(input: Tensor | TensorLike, classIndexOrLabel: number | string) {
        // convert to Tensor and add to TensorFlow model
        this.knnClassifier.addExample(asTensor(input), classIndexOrLabel);
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
     * @param {Tensor} input
     * @param {number} k
     */
    async classifyInternal(input: Tensor, k: number): Promise<KNNClassification> {
        const numClass = this.knnClassifier.getNumClasses();
        if (numClass <= 0) {
            throw new Error('There is no example in any class');
        } else {
            const res = await this.knnClassifier.predictClass(input, k);
            const {confidences: confidencesByLabel, classIndex, label} = res;
            const confidences = ObjectBuilder.from(confidencesByLabel).mapKeys(this.labelToIndex);
            return {
                classIndex,
                label,
                confidences,
                confidencesByLabel,
            }
        }
    }

    /**
     * Helper method converts a string label into its numeric classIndex
     * @param label
     * @private
     */
    private labelToIndex = (label: string): number => {
        // @ts-ignore accesses private property of the internal classifier.
        return this.knnClassifier.labelToClassId[label];
    }


    /**
     * Clear all examples in a label.
     * @param {number||string} labelOrIndex - The class index or label, a number or a string.
     */
    clearLabel(labelOrIndex: string | number) {
        this.knnClassifier.clearClass(labelOrIndex);
    }

    clearAllLabels(): void {
        this.knnClassifier.clearAllClasses();
    }

    /**
     * Get the example count for each label. It returns an object that maps class label to example count for each class.
     */
    getCountByLabel(): Record<string, number> {
        return this.knnClassifier.getClassExampleCount();
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
     * @param {string} [name] - Optional. The name of the JSON file that will be downloaded. e.g. "myKNN" or "myKNN.json".
     * If no fileName is provided, the default file name is "myKNN.json".
     */
    async save(name: string = 'myKNN.json'): Promise<void> {
        // get a dictionary of tensors keyed by label
        const rawDataset = this.getClassifierDataset();

        // add label to each entry
        const dataset = Object.entries(rawDataset).map(([label, tensor]) => ({
            ...tensor,
            label
        }));

        // get the values from each tensor
        const tensors = await Promise.all(
            Object.values(rawDataset).map((tensor) => tensor.data() )
        );

        // save the file
        const fileName = name.endsWith('.json') ? name : `${name}.json`;
        // TODO: is file type correct?
        await io.saveBlob(JSON.stringify({dataset, tensors}), fileName, 'application/octet-stream');
    }

    /**
     * Load a dataset from a JSON file. It's useful for restoring state.
     *
     * @param {string} pathOrData - The path for a valid JSON file.
     * @param {function} [callback] - Optional. A function to run once the dataset has been loaded.
     * If no callback is provided, it will return a promise that will be resolved once the dataset has loaded.
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
            // Note: needs to support previous downloads which used numeric keys instead of an array for dataset
            const keyedTensors = ObjectBuilder
                .fromValues(tensors.map((values, i) => tf.tensor<Rank.R2>(values, dataset[i].shape, dataset[i].dtype)))
                .createKeys((_, i) => dataset[i].label);
            this.knnClassifier.setClassifierDataset(keyedTensors);
        })(), callback);
    }
}

const KNNClassifier = () => new KNN();

export default KNNClassifier;
