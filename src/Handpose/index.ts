// Copyright (c) 2020 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* eslint prefer-destructuring: ["error", {AssignmentExpression: {array: false}}] */
/* eslint no-await-in-loop: "off" */

/*
 * Handpose: Palm detector and hand-skeleton finger tracking in the browser
 * Ported and integrated from all the hard work by: https://github.com/tensorflow/tfjs-models/tree/master/handpose
 */

import * as tf from "@tensorflow/tfjs";
import * as handposeCore from "@tensorflow-models/handpose";
import {EventEmitter} from "events";
import callCallback, {ML5Callback} from "../utils/callcallback";
import {extractImageElement, ImageArg, VideoArg} from "../utils/imageUtilities";
import {ArgSeparator} from "../utils/argSeparator";
import {createClass, createFactory} from "../utils/model-composition/MediaModel";

export interface HandposeOptions {
    flipHorizontal?: boolean;
    maxContinuousChecks?: number;
    detectionConfidence?: number;
    iouThreshold?: number;
    scoreThreshold?: number;
}

// AnnotatedPrediction is declared locally in ts handpose, but not exported
export type Unpromise<T> = T extends Promise<infer U> ? U : never;
type HandposePrediction = Unpromise<ReturnType<handposeCore.HandPose['estimateHands']>>;

class Handpose extends EventEmitter {
    ready: Promise<Handpose>;
    modelReady: boolean;
    video?: null | HTMLVideoElement;
    model: null | handposeCore.HandPose;
    config: HandposeOptions;

    /**
     * Create Handpose.
     * @param {HTMLVideoElement} video - An HTMLVideoElement.
     * @param {object} options - An object with options.
     * @param {function} callback - A callback to be called when the model is ready.
     */
    constructor(video?: HTMLVideoElement, options: HandposeOptions = {}, callback?: ML5Callback<Handpose>) {
        super();

        this.video = video || null;
        this.model = null;
        this.modelReady = false;
        this.config = options;

        this.ready = callCallback(this.loadModel(), callback);
    }

    /**
     * Load the model and set it to this.model
     * @return {this} the Handpose model.
     */
    async loadModel(): Promise<Handpose> {
        this.model = await handposeCore.load(this.config);
        this.modelReady = true;

        if (this.video && this.video.readyState === 0) {
            await new Promise<void>(resolve => {
                this.video!.onloadeddata = () => {
                    resolve();
                };
            });
        }

        // is this needed?
        this.predict();

        return this;
    }

    /**
     * Make a prediction based on a provided image or a frame of the current video
     * @return {Promise<HandposePrediction>} a prediction
     */
    async predict(inputOr?: ImageArg, callback?: (result: HandposePrediction) => void): Promise<HandposePrediction> {
        const input = extractImageElement(inputOr) ?? this.video;
        if (!input) {
            // throw error?
            return [];
        }
        const {flipHorizontal} = this.config;
        const result = await this.model.estimateHands(input, flipHorizontal);
        this.emit("predict", result);

        if (this.video) {
            // should we do this even if an image was provided?
            return tf.nextFrame().then(() => this.predict());
        }

        if (typeof callback === "function") {
            callback(result);
        }

        return result;
    }
}

const handpose = (videoOrOptionsOrCallback: VideoArg | HandposeOptions | ML5Callback<Handpose>, optionsOrCallback: HandposeOptions | ML5Callback<Handpose>, cb?: ML5Callback<Handpose>) => {
    const {options, video, callback} = new ArgSeparator(videoOrOptionsOrCallback, optionsOrCallback, cb);
    const instance = new Handpose(video, options, callback);
    return callback ? instance : instance.ready;
};

const loadModel = (config: HandposeOptions): Promise<handposeCore.HandPose> => handposeCore.load(config)

const BaseClass = createClass(loadModel, {} as Partial<HandposeOptions>);

class HandposeClass extends BaseClass {
    predict = this._makeImageMethod(
        async (image) => {
            const {flipHorizontal} = this.config;
            return await this.instance!.estimateHands(image, flipHorizontal);
        },
        "hand"
    )
}

const factory = createFactory(HandposeClass);

const x = async () => {
  const instance = await factory();
  const p = await instance.predict();
}

export default handpose;
