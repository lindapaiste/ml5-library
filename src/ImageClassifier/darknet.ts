// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tf from "@tensorflow/tfjs";
import {LayersModel, Tensor, Tensor4D} from "@tensorflow/tfjs";
import {Classification, getTopKClassesFromTensor} from "../utils/gettopkclasses";
import IMAGENET_CLASSES_DARKNET from "../utils/IMAGENET_CLASSES_DARKNET";
import {TfImageSource} from "../utils/imageUtilities";
import {toTensor} from "../utils/imageConversion";
import {Tensor3D} from "@tensorflow/tfjs-core";

// TODO: try to combine logic from darknet and doodlenet into one class.
// The differences can be implemented via different arguments passed to the constructor

/**
 * Shared interface for a classifier
 */
export interface ClassifierModel {
    //readonly model: LayersModel;
    //warmup(): Promise<void>;
    classify(img: TfImageSource, topk: number): Promise<Classification[]>;
}

export interface ClassifierModelConfig {
    /**
     * Path to the model.json file.
     */
    url: string;
    /**
     * Expected width/height of the image for this model.
     */
    imgSize: number;
    /**
     * Class names keyed by index.
     */
    classes: string[] | Record<number, string>;
}

const SETTINGS = {
    // this a 28mb model
    reference: {
        url: "https://cdn.jsdelivr.net/gh/ml5js/ml5-data-and-models@master/models/darknetclassifier/darknetreference/model.json",
        imgSize: 256,
        classes: IMAGENET_CLASSES_DARKNET,
    },
    // this a 4mb model
    tiny: {
        url: "https://cdn.jsdelivr.net/gh/ml5js/ml5-data-and-models@master/models/darknetclassifier/darknettiny/model.json",
        imgSize: 224,
        classes: IMAGENET_CLASSES_DARKNET,
    }
}

/**
 * Reformat each image before prediction.
 *
 * @param img
 * @param size
 */
function preProcess(img: TfImageSource, size: number): Tensor4D {
    const image = toTensor(img);
    // note: rank gets lost here
    const normalized: Tensor3D = image.toFloat().div(tf.scalar(255));
    let resized = normalized;
    if (normalized.shape[0] !== size || normalized.shape[1] !== size) {
        const alignCorners = true;
        resized = tf.image.resizeBilinear(normalized, [size, size], alignCorners);
    }
    // batched
    return resized.reshape([1, size, size, 3]);
}

/**
 * The fully set-up and initialized model
 */
export class Darknet implements ClassifierModel {
    constructor(public readonly model: LayersModel, private config: ClassifierModelConfig) {
    }

    async warmup(): Promise<void> {
        // Warmup the model.
        const result = tf.tidy(() => this.model.predict(
            tf.zeros([1, this.config.imgSize, this.config.imgSize, 3])
        ) as Tensor);
        await result.data();
        result.dispose();
    }

    async classify(img: TfImageSource, topk: number = 3): Promise<Classification[]> {
        const logits = tf.tidy(() => {
            const imgData = preProcess(img, this.config.imgSize);
            const predictions = this.model.predict(imgData) as Tensor;
            // TODO: why does darknet use softmax and doodlenet doesn't?
            return tf.softmax(predictions);
        });
        const classes = await getTopKClassesFromTensor(logits, topk, this.config.classes);
        logits.dispose();
        return classes;
    }
}

export const load = async (version: string | number): Promise<ClassifierModel> => {
    if (version !== "reference" && version !== "tiny") {
        throw new Error("Please select a Darknet version.  Must be one of 'reference' or 'tiny'.");
    }

    const config = SETTINGS[version];
    const model = await tf.loadLayersModel(config.url);
    const darknet = new Darknet(model, config);
    await darknet.warmup();
    return darknet;
}
