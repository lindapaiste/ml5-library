// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Image Classifier using pre-trained networks
*/

import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as darknet from "./darknet";
import {ClassifierModel} from "./darknet";
import * as doodlenet from "./doodlenet";
import * as customModel from "./customModel";
import callCallback, {Callback} from "../utils/callcallback";
import {imgToTensor, TfImageSource} from "../utils/imageUtilities";
import {ArgSeparator} from "../utils/argSeparator";

const MOBILENET_DEFAULTS: mobilenet.ModelConfig = {
    version: 2,
    alpha: 1.0,
}
const DEFAULTS = {
    topk: 3,
};

const IMAGE_SIZE = 224;
const MODEL_OPTIONS = ["mobilenet", "darknet", "darknet-tiny", "doodlenet"];

interface Options {
    version?: number | string;
    alpha?: number; // used by MobileNet only
    topk: number;
}

export interface LabelAndConfidence {
    label: string;
    confidence: number;
}

/**
 * Load the correct ClassifierModel instance based on the name.
 * @param modelNameOrUrl
 * @param options
 */
const loadClassifier = async (modelNameOrUrl: string, options: Partial<Options> = {}): Promise<ClassifierModel> => {
    switch (modelNameOrUrl) {
        case "mobilenet":
            // TODO validate version -- but first check tensorflow errors
            // @ts-ignore
            return await mobilenet.load({...MOBILENET_DEFAULTS, ...options});
        case "darknet-tiny":
            return await darknet.load("tiny");
        case "darknet":
            // can also pass version through options
            return await darknet.load(options.version || "reference");
        case "doodlenet":
            return await doodlenet.load();
        default:
            // assume that all other names are URLs
            return await customModel.load(modelNameOrUrl);
    }
}

class ImageClassifier {

    constructor(public readonly model: ClassifierModel, public config: Options, public video?: HTMLVideoElement) {
    }

    /**
     * Classifies the given input and returns an object with labels and confidence
     * @param {HTMLImageElement | HTMLCanvasElement | HTMLVideoElement} imgToPredict -
     *    takes an image to run the classification on.
     * @param {number} numberOfClasses - a number of labels to return for the image
     *    classification.
     * @return {object} an object with {label, confidence}.
     */
    async classifyInternal(imgToPredict: TfImageSource, numberOfClasses: number): Promise<LabelAndConfidence[]> {
        // TODO: is this needed?
        await tf.nextFrame();

        // TODO: move to utilities
        if (imgToPredict instanceof HTMLVideoElement && imgToPredict.readyState === 0) {
            const video = imgToPredict;
            // Wait for the video to be ready
            await new Promise<void>(resolve => {
                video.onloadeddata = () => resolve();
            });
        }

        // Process the images
        // TODO: It seems like this isn't needed because models handle the resize themselves.  But could move it here.
        const processedImg = imgToTensor(imgToPredict, [IMAGE_SIZE, IMAGE_SIZE]);
        const classes = await this.model.classify(processedImg, numberOfClasses);
        processedImg.dispose();

        // TODO: why is this reformatted?  Might be better to keep consistency.
        return classes.map(c => ({label: c.className, confidence: c.probability}));
    }

    /**
     * Classifies the given input and takes a callback to handle the results
     * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} inputNumOrCallback -
     *    takes any of the following params
     * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} numOrCallback -
     *    takes any of the following params
     * @param {function} cb - a callback function that handles the results of the function.
     * @return {function} a promise or the results of a given callback, cb.
     */
    async classify(
        inputNumOrCallback: TfImageSource | number | Callback<LabelAndConfidence[]>,
        numOrCallback: number | Callback<LabelAndConfidence[]>,
        cb?: Callback<LabelAndConfidence[]>
    ) {
        const {
            image: imgToPredict,
            number: numberOfClasses = this.config.topk,
            callback
        } = new ArgSeparator(this.video, inputNumOrCallback, numOrCallback, cb);

        if (!(imgToPredict)) {
            // Handle unsupported input
            throw new Error(
                "No input image provided. If you want to classify a video, pass the video element in the constructor.",
            );
        }

        return callCallback(this.classifyInternal(imgToPredict, numberOfClasses), callback);
    }

    /**
     * Will be deprecated soon in favor of ".classify()" - does the same as .classify()
     * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} inputNumOrCallback - takes any of the following params
     * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} numOrCallback - takes any of the following params
     * @param {function} cb - a callback function that handles the results of the function.
     * @return {function} a promise or the results of a given callback, cb.
     */
    async predict (
        inputNumOrCallback: TfImageSource | number | Callback<LabelAndConfidence[]>,
        numOrCallback: number | Callback<LabelAndConfidence[]>,
        cb?: Callback<LabelAndConfidence[]>
    ) {
        return this.classify(inputNumOrCallback, numOrCallback, cb);
    }
}

/**
 * Create an ImageClassifier.
 * @param {string} model - The name or the URL of the model to use. Current model name options
 *    are: 'mobilenet', 'darknet', 'darknet-tiny', and 'doodlenet'.
 * @param {HTMLVideoElement} [videoOrOptionsOrCallback] - An HTMLVideoElement.
 * @param {object} [optionsOrCallback] - An object with options.
 * @param {function} [cb] - A callback to be called when the model is ready.
 */
const imageClassifier = (
    model: string,
    videoOrOptionsOrCallback?: HTMLVideoElement | Options | Callback<ImageClassifier>,
    optionsOrCallback?: Options | Callback<ImageClassifier>,
    cb?: Callback<ImageClassifier>
) => {
    const {string: modelName, video, options, callback} = new ArgSeparator(model, videoOrOptionsOrCallback, optionsOrCallback, cb);

    if (modelName === undefined) {
        throw new Error('Please specify a model to use. E.g: "MobileNet"');
    }

    const load = async () => {
        const model = await loadClassifier(modelName, options);
        const config = {
            ...DEFAULTS,
            ...options,
        }
        return new ImageClassifier(model, config, video);
    }

    return callCallback(load(), callback);
};

export default imageClassifier;
