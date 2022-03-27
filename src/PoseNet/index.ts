// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
PoseNet
The original PoseNet model was ported to TensorFlow.js by Dan Oved.
*/

import {EventEmitter} from 'events';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import callCallback, {Callback} from '../utils/callcallback';
import {ArgSeparator} from "../utils/argSeparator";
import {Dimensions} from "../utils/dimensions";
import {
    MobileNetMultiplier,
    PoseNetArchitecture,
    PoseNetDecodingMethod,
    PoseNetOutputStride,
    PoseNetQuantBytes
} from "@tensorflow-models/posenet/dist/types";
import {TfImageSource, VideoArg, videoLoaded} from "../utils/imageUtilities";

/**
 * @typedef {Object} PoseNetOptions
 * @property {string} architecture - default 'MobileNetV1',
 * @property {number} inputResolution - default 257,
 * @property {number} outputStride - default 16
 * @property {boolean} flipHorizontal - default false
 * @property {number} minConfidence - default 0.5
 * @property {number} maxPoseDetections - default 5
 * @property {number} scoreThreshold - default 0.5
 * @property {number} nmsRadius - default 20
 * @property {string} detectionType - default 'single'
 * @property {number} nmsRadius - default 0.75,
 * @property {number} quantBytes - default 2,
 * @property {(string | null)} modelUrl - default null
 * @property {number} multiplier - default 0.75 (only for MobileNet)
 */
interface PoseNetOptions {
    architecture: PoseNetArchitecture;
    outputStride: PoseNetOutputStride;
    flipHorizontal: boolean;
    minConfidence: number;
    maxPoseDetections: number;
    scoreThreshold: number;
    nmsRadius: number;
    detectionType: PoseNetDecodingMethod;
    inputResolution: number | Dimensions;
    multiplier: MobileNetMultiplier;
    quantBytes: PoseNetQuantBytes;
    modelUrl?: string;
}

const DEFAULTS: PoseNetOptions = {
    architecture: 'MobileNetV1', // 'MobileNetV1', 'ResNet50'
    outputStride: 16, // 8, 16, 32
    flipHorizontal: false, // true, false
    minConfidence: 0.5,
    maxPoseDetections: 5, // any number > 1
    scoreThreshold: 0.5,
    nmsRadius: 20, // any number > 0
    detectionType: 'multi-person', // 'single-person'
    inputResolution: 256, // or { width: 257, height: 200 }
    multiplier: 0.75, // 1.0, 0.75, or 0.50 -- only for MobileNet
    quantBytes: 2, // 4, 2, 1
    modelUrl: undefined, // url path to model
};

// results are re-formatted from the PoseNet types
interface ResultKeypoint {
    x: number;
    y: number;
    confidence: number;
}

// parts are stored as top-level properties, which makes for difficult typing
interface ResultPose {
    keypoints: posenet.Keypoint[];
    // TODO: why rename score to confidence in keypoints but not here?
    score: number;
    [s: string]: ResultKeypoint | number | posenet.Keypoint[];
}

interface Result {
    pose: ResultPose;
    skeleton: posenet.Keypoint[][];
}

//TODO: is it ok to move configuration options from top-level properties to this.config?

class PoseNet extends EventEmitter implements PoseNetOptions {

    video?: HTMLVideoElement;
    ready: Promise<PoseNet>;
    net?: posenet.PoseNet;
    // config
    architecture: PoseNetArchitecture;
    outputStride: PoseNetOutputStride;
    flipHorizontal: boolean;
    minConfidence: number;
    maxPoseDetections: number;
    scoreThreshold: number;
    nmsRadius: number;
    detectionType: PoseNetDecodingMethod;
    inputResolution: number | Dimensions;
    multiplier: MobileNetMultiplier;
    quantBytes: PoseNetQuantBytes;
    modelUrl?: string;

    /**
     * Create a PoseNet model.
     * @param {HTMLVideoElement} [video] - Optional. A HTML video element or a p5 video element.
     * @param {Partial<PoseNetOptions>} [options] - Optional. An object describing a model accuracy and performance.
     * @param {string} [detectionType] - Optional. A String value to run 'single' or 'multiple' estimation.
     * @param {function} [callback] - Optional. A function to run once the model has been loaded.
     *    If no callback is provided, it will return a promise that will be resolved once the
     *    model has loaded.
     */
    constructor(video?: HTMLVideoElement, options: Partial<PoseNetOptions> = {}, detectionType?: string, callback?: Callback<PoseNet>) {
        super();
        this.video = video;
        /**
         * The type of detection. 'single' or 'multiple'
         * @type {string}
         * @public
         */
        this.detectionType = toDetectionType(detectionType || options.detectionType) || DEFAULTS.detectionType;
        this.modelUrl = options.modelUrl;
        this.architecture = options.architecture || DEFAULTS.architecture;
        this.outputStride = options.outputStride || DEFAULTS.outputStride;
        this.flipHorizontal = options.flipHorizontal || DEFAULTS.flipHorizontal;
        this.scoreThreshold = options.scoreThreshold || DEFAULTS.scoreThreshold;
        this.minConfidence = options.minConfidence || DEFAULTS.minConfidence;
        this.maxPoseDetections = options.maxPoseDetections || DEFAULTS.maxPoseDetections;
        this.multiplier = options.multiplier || DEFAULTS.multiplier;
        this.inputResolution = options.inputResolution || DEFAULTS.inputResolution;
        this.quantBytes = options.quantBytes || DEFAULTS.quantBytes;
        this.nmsRadius = options.nmsRadius || DEFAULTS.nmsRadius;
        this.ready = callCallback(this.load(), callback);
        // this.then = this.ready.then;
    }

    async load(): Promise<this> {
        this.net = await posenet.load(this);

        if (this.video) {
            await videoLoaded(this.video);
            this.pose(this.detectionType);
        }
        return this;
    }

    private skeleton(keypoints: posenet.Keypoint[], confidence = this.minConfidence): posenet.Keypoint[][] {
        return posenet.getAdjacentKeyPoints(keypoints, confidence);
    }

    private formatResult(pose: posenet.Pose): Result {
        return {
            // add each part as a keyed property on the pose
            pose: pose.keypoints.reduce((obj: ResultPose, keypoint) => ({
                ...obj,
                [keypoint.part]: {
                    x: keypoint.position.x,
                    y: keypoint.position.y,
                    confidence: keypoint.score,
                }
            }), pose),
            // include skeleton
            skeleton: this.skeleton(pose.keypoints),
        };
    }

    // Minimize the split logic between single and multiple pose detection.  Share as much code as possible
    private async rawResult (mode: PoseNetDecodingMethod = this.detectionType, image: TfImageSource): Promise<posenet.Pose[]> {
        if ( mode === "single-person") {
            const pose = await this.net!.estimateSinglePose(image, {flipHorizontal: this.flipHorizontal});
            return [pose];
        } else {
            return this.net!.estimateMultiplePoses(image, {
                flipHorizontal: this.flipHorizontal,
                maxDetections: this.maxPoseDetections,
                scoreThreshold: this.scoreThreshold,
                nmsRadius: this.nmsRadius
            });
        }
    }

    /**
     * Given an image or video, returns an array of objects containing pose estimations
     *    using single or multi-pose detection.
     * @param {PoseNetDecodingMethod} mode
     * @param {HTMLVideoElement || p5.Video || function} inputOr
     * @param {function} cb
     */
    async pose(mode: PoseNetDecodingMethod = this.detectionType, inputOr?: TfImageSource | Callback<Result[]>, cb?: Callback<Result[]>): Promise<Result[]> {
        const {image, callback} = new ArgSeparator(this.video, inputOr, cb);

        await this.ready;
        const poses = await this.rawResult(mode, image);
        const result = poses.map(pose => this.formatResult(pose));
        this.emit('pose', result);

        if (this.video) {
            return tf.nextFrame().then(() => this.pose(mode));
        }

        // TODO: proper callback handling
        if (typeof callback === 'function') {
            callback(result);
        }

        return result;
    }

    /**
     * Given an image or video, returns an array of objects containing pose estimations
     *    using single-pose detection.
     * @param {HTMLVideoElement || p5.Video || function} inputOr
     * @param {function} cb
     */
    async singlePose(inputOr?: TfImageSource | Callback<Result[]>, cb?: Callback<Result[]>): Promise<Result[]> {
        return this.pose("single-person", inputOr, cb);
    }

    /**
     * Given an image or video, returns an array of objects containing pose estimations
     *    using multi-pose detection.
     * @param {HTMLVideoElement || p5.Video || function} inputOr
     * @param {function} cb
     */
    async multiPose(inputOr?: TfImageSource | Callback<Result[]>, cb?: Callback<Result[]>): Promise<Result[]> {
        return this.pose("multi-person", inputOr, cb);
    }
}

const toDetectionType = (text: string | undefined): PoseNetDecodingMethod | undefined => {
    // look for any string that contains single or multiple, case-insensitive
    if (text?.match(/single/i)) {
        return "single-person";
    } else if (text?.match(/multi/i)) {
        return "multi-person";
    }
    // only throw error if there was a string but no match
    if (text) {
        throw new Error(`Invalid detection type ${text}`);
    }
}

// TODO: should this accept four arguments? Seems to accept detectionType string.
const poseNet = (videoOrOptionsOrCallback?: VideoArg | PoseNetOptions | Callback<PoseNet> | string, optionsOrCallback?: PoseNetOptions | Callback<PoseNet> | string, cb?: Callback<PoseNet> | string) => {
    const {string, video, options, callback} = new ArgSeparator(videoOrOptionsOrCallback, optionsOrCallback, cb);
    return new PoseNet(video, options, string, callback);
};

export default poseNet;
