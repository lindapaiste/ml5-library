// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT


import {Tensor} from "@tensorflow/tfjs";

/**
 * @typedef {Object} Classification
 * @param {string} className - The name of the matched class.
 * @param {number} probability - The probability from 0 to 1 that this class name is a match.
 */
export interface Classification {
    className: string;
    probability: number;
}


/**
 * Utility function which limits classifier predictions to a provided number `topK`
 *
 * @param {number[] | Float32Array} values - Array of probabilities
 * where the indexes match the indexes of the provided classes
 * and the values of the probability that that class is a match.
 * @param {number} k - Number of top classes to return.
 * @param {string[]} CLASSES - Array of labels.
 * @return {Classification[]} - Array of objects with properties probability and className.
 */
export function getTopKClassesFromArray(values: number[] | Float32Array, k: number, CLASSES: string[] | Record<number, string>): Classification[] {
    const labeled = [...values].map((value, i) => ({
        className: CLASSES[i],
        probability: value
    }));
    // note: the performance would be better if we don't sort the whole array
    // we only need the top k, so the order of others doesn't matter.
    labeled.sort((a, b) => b.probability - a.probability);
    return labeled.slice(0, k);
}

/**
 * @param {Tensor} logits - Tensorflow Tensor, likely of type "float32"
 * @param {number} k - Number of top classes to return.
 * @param {string[]} CLASSES - Array of labels.
 * @return {Promise<Classification[]>} - Array of objects with properties probability and className.
 */
export async function getTopKClassesFromTensor(logits: Tensor, k: number, CLASSES: string[] | Record<number, string>): Promise<Classification[]> {
    const topK = await logits.topk(k);
    /**
     * @type Float32Array
     */
    const values = await topK.values.data();
    topK.values.dispose();
    /**
     * @type Int32Array
     */
    const indices = await topK.indices.data();
    topK.indices.dispose();
    // note cannot map the values directly because Float32Array can only map to another Float32Array
    return [...values].map((probability, i) => ({
        className: CLASSES[indices[i]],
        probability,
    }));
}

export interface LabelAndConfidence {
    label: string;
    confidence: number;
}

export const toLabelAndConfidence = ({className, probability}: Classification): LabelAndConfidence => ({
    label: className,
    confidence: probability
})

export default {getTopKClassesFromArray, getTopKClassesFromTensor};
