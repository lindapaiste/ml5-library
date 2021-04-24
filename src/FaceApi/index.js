// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* eslint no-await-in-loop: "off" */

/*
 * FaceApi: real-time face recognition, and landmark detection
 * Ported and integrated from all the hard work by: https://github.com/justadudewhohacks/face-api.js?files=1
 */

import * as tf from "@tensorflow/tfjs";
import * as faceapi from "face-api.js";
import callCallback from "../utils/callcallback";
import {ImageModelArgs} from "../utils/imageModelArgs";

/**
 * Settings for the combined model.
 *
 * @typedef {Object} FaceApiBasicOptions
 * @property {number} minConfidence
 * @property {boolean} withLandmarks
 * @property {boolean} withDescriptors
 * @property {boolean} withTinyNet
 * @property {boolean} withFaceExpressions
 * @property {boolean} withAgeAndGender
 */

/**
 * An object of paths to JSON models for various face properties.
 *
 * @typedef {Object} FaceApiModelUrls
 * @property {string} Mobilenetv1Model
 * @property {string} TinyFaceDetectorModel
 * @property {string} FaceLandmarkModel
 * @property {string} FaceLandmark68TinyNet
 * @property {string} FaceRecognitionModel
 */

/**
 * @type {FaceApiModelUrls}
 */
const MODEL_URLS = {
    Mobilenetv1Model:
        "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/ssd_mobilenetv1_model-weights_manifest.json",
    TinyFaceDetectorModel:
        "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/tiny_face_detector_model-weights_manifest.json",
    FaceLandmarkModel:
        "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_landmark_68_model-weights_manifest.json",
    FaceLandmark68TinyNet:
        "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_landmark_68_tiny_model-weights_manifest.json",
    FaceRecognitionModel:
        "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_recognition_model-weights_manifest.json",
}
const MODEL_NAMES = Object.keys(MODEL_URLS);

/**
 * this.config property stores all models nested under property key `MODEL_URLS`.
 *
 * @typedef {Object} HasModelUrls
 * @property {FaceApiModelUrls} MODEL_URLS
 * @typedef {FaceApiBasicOptions & HasModelUrls} FaceApiConfig
 */

/**
 * options argument expects the models to be top-level properties.
 *
 * @typedef {FaceApiBasicOptions & FaceApiModelUrls} FaceApiOptions
 */

/**
 * @type FaceApiConfig
 */
const DEFAULTS = {
    withLandmarks: true,
    withDescriptors: true,
    minConfidence: 0.5,
    withTinyNet: true,
    withFaceExpressions: false,
    withAgeAndGender: false,
    MODEL_URLS: {
        Mobilenetv1Model:
            "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/ssd_mobilenetv1_model-weights_manifest.json",
        TinyFaceDetectorModel:
            "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/tiny_face_detector_model-weights_manifest.json",
        FaceLandmarkModel:
            "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_landmark_68_model-weights_manifest.json",
        FaceLandmark68TinyNet:
            "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_landmark_68_tiny_model-weights_manifest.json",
        FaceRecognitionModel:
            "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_recognition_model-weights_manifest.json",
    }
};

/**
 * @property {HTMLVideoElement | null } video - Video element, if provided
 * @property {faceapi} model - Model from face-api.js package
 * @property {FaceApiConfig} config - Configuration options for the model
 * @property {boolean} modelReady -
 */
class FaceApiBase {
    /**
     * Create FaceApi.
     * @param {HTMLVideoElement} [video] - An HTMLVideoElement.
     * @param {object} [options] - An object with options.
     * @param {function} [callback] - A callback to be called when the model is ready.
     */
    constructor(video, options, callback) {
        this.video = video || null;
        this.model = faceapi;
        this.modelReady = false;
        this.detectorOptions = null;
        this.config = {
            minConfidence: this.checkUndefined(options.minConfidence, DEFAULTS.minConfidence),
            withLandmarks: this.checkUndefined(options.withLandmarks, DEFAULTS.withLandmarks),
            withDescriptors: this.checkUndefined(options.withDescriptors, DEFAULTS.withDescriptors),
            withTinyNet: this.checkUndefined(options.withTinyNet, DEFAULTS.withTinyNet),
            MODEL_URLS: {
                Mobilenetv1Model: this.checkUndefined(
                    options.Mobilenetv1Model,
                    DEFAULTS.MODEL_URLS.Mobilenetv1Model,
                ),
                TinyFaceDetectorModel: this.checkUndefined(
                    options.TinyFaceDetectorModel,
                    DEFAULTS.MODEL_URLS.TinyFaceDetectorModel,
                ),
                FaceLandmarkModel: this.checkUndefined(
                    options.FaceLandmarkModel,
                    DEFAULTS.MODEL_URLS.FaceLandmarkModel,
                ),
                FaceLandmark68TinyNet: this.checkUndefined(
                    options.FaceLandmark68TinyNet,
                    DEFAULTS.MODEL_URLS.FaceLandmark68TinyNet,
                ),
                FaceRecognitionModel: this.checkUndefined(
                    options.FaceRecognitionModel,
                    DEFAULTS.MODEL_URLS.FaceRecognitionModel,
                ),
            },
        };

        this.ready = callCallback(this.loadModel(), callback);
    }

    /**
     * Load the model and set it to this.model
     * @return {this}
     */
    async loadModel() {
        Object.keys(this.config.MODEL_URLS).forEach(item => {
            if (MODEL_NAMES.includes(item)) {
                this.config.MODEL_URLS[item] = this.getModelPath(this.config.MODEL_URLS[item]);
            }
        });

        const {
            Mobilenetv1Model,
            TinyFaceDetectorModel,
            FaceLandmarkModel,
            FaceRecognitionModel,
            FaceLandmark68TinyNet,
        } = this.config.MODEL_URLS;

        this.model = faceapi;


        if (this.config.withTinyNet === true) {
            this.detectorOptions = new faceapi.TinyFaceDetectorOptions({
                // TODO: property name minConfidence is not supported -- should this be scoreThreshold?
                minConfidence: this.config.minConfidence,
                inputSize: 512,
            });
        } else {
            this.detectorOptions = new faceapi.SsdMobilenetv1Options({
                minConfidence: this.config.minConfidence,
            });
        }

        // check which model to load - tiny or normal
        if (this.config.withTinyNet === true) {
            await this.model.loadFaceLandmarkTinyModel(FaceLandmark68TinyNet);
            await this.model.loadTinyFaceDetectorModel(TinyFaceDetectorModel);
        } else {
            await this.model.loadFaceLandmarkModel(FaceLandmarkModel);
            await this.model.loadSsdMobilenetv1Model(Mobilenetv1Model);
        }
        await this.model.loadFaceRecognitionModel(FaceRecognitionModel);

        this.modelReady = true;
        return this;
    }

    /**
     * .detect() - classifies multiple features by default
     * @param {*} optionsOrCallback
     * @param {*} configOrCallback
     * @param {*} cb
     */
    async detect(optionsOrCallback, configOrCallback, cb) {
        const {image, options = {}, callback} = new ImageModelArgs(optionsOrCallback, configOrCallback, cb);

        const imgToClassify = image || this.video;
        if (!imgToClassify) {
            // Handle unsupported input
            throw new Error(
                "No input image provided. If you want to classify a video, pass the video element in the constructor.",
            );
        }

        return callCallback(this.detectInternal(imgToClassify, options, false), callback);
    }

    /**
     * Internal function handles detecting single or multiple faces
     * @param {(HTMLVideoElement | HTMLImageElement | HTMLCanvasElement)} imgToClassify
     * @param {Partial<FaceApiBasicOptions>} faceApiOptions
     * @param {boolean} single
     */
    async detectInternal(imgToClassify, faceApiOptions, single) {
        await this.ready;
        await tf.nextFrame();

        if (this.video && this.video.readyState === 0) {
            await new Promise(resolve => {
                this.video.onloadeddata = () => resolve();
            });
        }

        // merge any options provided for this detection with those set on the class instance
        const config = this.setReturnOptions(faceApiOptions);

        // build the task based on the settings
        const detect = single ? this.model.detectSingleFace : this.model.detectAllFaces;
        let task = detect(imgToClassify, this.detectorOptions);
        if (config.withLandmarks) {
            task = task.withFaceLandmarks(config.withTinyNet);
        }
        if (config.withDescriptors) {
            task = task.withFaceDescriptors();
        }
        if (config.withAgeAndGender) {
            task = task.withAgeAndGender();
        }
        if (config.withFaceExpressions) {
            task = task.withFaceExpressions();
        }

        // get the result
        let result = await task.run();
        // always resize the results to the input image size
        result = this.resizeResults(result, imgToClassify.width, imgToClassify.height);
        // assign the {parts} object after resizing
        result = this.landmarkParts(result);

        return result;
    }

    /**
     * .detecSinglet() - classifies a single feature with higher accuracy
     * @param {*} optionsOrCallback
     * @param {*} configOrCallback
     * @param {*} cb
     */
    async detectSingle(imageOrOptionsOrCallback, optionsOrCallback, cb) {
        const {image, options, callback} = new ImageModelArgs(imageOrOptionsOrCallback, optionsOrCallback, cb);
        let imgToClassify = this.video;
        let callback;
        let faceApiOptions = this.config;

        // Handle the image to predict
        if (typeof optionsOrCallback === "function") {
            imgToClassify = this.video;
            callback = optionsOrCallback;
            // clean the following conditional statement up!
        } else if (
            optionsOrCallback instanceof HTMLImageElement ||
            optionsOrCallback instanceof HTMLCanvasElement ||
            optionsOrCallback instanceof HTMLVideoElement ||
            optionsOrCallback instanceof ImageData
        ) {
            imgToClassify = optionsOrCallback;
        } else if (
            typeof optionsOrCallback === "object" &&
            (optionsOrCallback.elt instanceof HTMLImageElement ||
                optionsOrCallback.elt instanceof HTMLCanvasElement ||
                optionsOrCallback.elt instanceof HTMLVideoElement ||
                optionsOrCallback.elt instanceof ImageData)
        ) {
            imgToClassify = optionsOrCallback.elt; // Handle p5.js image
        } else if (
            typeof optionsOrCallback === "object" &&
            optionsOrCallback.canvas instanceof HTMLCanvasElement
        ) {
            imgToClassify = optionsOrCallback.canvas; // Handle p5.js image
        } else if (!(this.video instanceof HTMLVideoElement)) {
            // Handle unsupported input
            throw new Error(
                "No input image provided. If you want to classify a video, pass the video element in the constructor. ",
            );
        }

        if (typeof configOrCallback === "object") {
            faceApiOptions = configOrCallback;
        } else if (typeof configOrCallback === "function") {
            callback = configOrCallback;
        }

        if (typeof cb === "function") {
            callback = cb;
        }

        return callCallback(this.detectSingleInternal(imgToClassify, faceApiOptions), callback);
    }

    /**
     * Detects only a single feature
     * @param {HTMLImageElement || HTMLVideoElement} imgToClassify
     * @param {Object} faceApiOptions
     */
    async detectSingleInternal(imgToClassify, faceApiOptions) {
        await this.ready;
        await tf.nextFrame();

        if (this.video && this.video.readyState === 0) {
            await new Promise(resolve => {
                this.video.onloadeddata = () => resolve();
            });
        }

        // sets the return options if any are passed in during .detect() or .detectSingle()
        this.config = this.setReturnOptions(faceApiOptions);

        const {withLandmarks, withDescriptors} = this.config;

        let result;
        if (withLandmarks) {
            if (withDescriptors) {
                result = await this.model
                    .detectSingleFace(imgToClassify, this.detectorOptions)
                    .withFaceLandmarks(this.config.withTinyNet)
                    .withFaceDescriptor();
            } else {
                result = await this.model
                    .detectSingleFace(imgToClassify, this.detectorOptions)
                    .withFaceLandmarks(this.config.withTinyNet);
            }
        } else if (!withLandmarks) {
            result = await this.model.detectSingleFace(imgToClassify);
        } else {
            result = await this.model
                .detectSingleFace(imgToClassify, this.detectorOptions)
                .withFaceLandmarks(this.config.withTinyNet)
                .withFaceDescriptor();
        }

        // always resize the results to the input image size
        result = this.resizeResults(result, imgToClassify.width, imgToClassify.height);

        // assign the {parts} object after resizing
        result = this.landmarkParts(result);

        return result;
    }

    /**
     * Check if the given _param is undefined, otherwise return the _default
     * @param {*} _param
     * @param {*} _default
     */
    checkUndefined(_param, _default) {
        return _param !== undefined ? _param : _default;
    }

    /**
     * Checks if the given string is an absolute or relative path and returns
     *      the path to the modelJson
     * @param {String} absoluteOrRelativeUrl
     */
    getModelPath(absoluteOrRelativeUrl) {
        const modelJsonPath = this.isAbsoluteURL(absoluteOrRelativeUrl)
            ? absoluteOrRelativeUrl
            : window.location.pathname + absoluteOrRelativeUrl;
        return modelJsonPath;
    }

    /**
     * Sets the return options for .detect() or .detectSingle() in case any are given
     * @param {Object} faceApiOptions
     */
    setReturnOptions(faceApiOptions) {
        const output = Object.assign({}, this.config);
        const options = ["withLandmarks", "withDescriptors"];

        options.forEach(prop => {
            if (faceApiOptions[prop] !== undefined) {
                this.config[prop] = faceApiOptions[prop];
            } else {
                output[prop] = this.config[prop];
            }
        });

        return output;
    }

    /**
     * Resize results to size of input image
     * @param {*} str
     */
    resizeResults(detections, width, height) {
        if (width === undefined || height === undefined) {
            throw new Error("width and height must be defined");
        }
        return this.model.resizeResults(detections, {
            width,
            height
        });
    }

    /* eslint class-methods-use-this: "off" */
    isAbsoluteURL(str) {
        const pattern = new RegExp("^(?:[a-z]+:)?//", "i");
        return !!pattern.test(str);
    }

    /**
     * get parts from landmarks
     * @param {*} result
     */
    landmarkParts(result) {
        let output;
        // multiple detections is an array
        if (Array.isArray(result) === true) {
            output = result.map(item => {
                // if landmarks exist return parts
                const newItem = Object.assign({}, item);
                if (newItem.landmarks) {
                    const {landmarks} = newItem;
                    newItem.parts = {
                        mouth: landmarks.getMouth(),
                        nose: landmarks.getNose(),
                        leftEye: landmarks.getLeftEye(),
                        leftEyeBrow: landmarks.getLeftEyeBrow(),
                        rightEye: landmarks.getRightEye(),
                        rightEyeBrow: landmarks.getRightEyeBrow(),
                        jawOutline: landmarks.getJawOutline(),
                    };
                } else {
                    newItem.parts = {
                        mouth: [],
                        nose: [],
                        leftEye: [],
                        leftEyeBrow: [],
                        rightEye: [],
                        rightEyeBrow: [],
                        jawOutline: [],
                    };
                }
                return newItem;
            });
            // single detection is an object
        } else {
            output = Object.assign({}, result);
            if (output.landmarks) {
                const {landmarks} = result;
                output.parts = {
                    mouth: landmarks.getMouth(),
                    nose: landmarks.getNose(),
                    leftEye: landmarks.getLeftEye(),
                    leftEyeBrow: landmarks.getLeftEyeBrow(),
                    rightEye: landmarks.getRightEye(),
                    rightEyeBrow: landmarks.getRightEyeBrow(),
                };
            } else {
                output.parts = {
                    mouth: [],
                    nose: [],
                    leftEye: [],
                    leftEyeBrow: [],
                    rightEye: [],
                    rightEyeBrow: [],
                };
            }
        }

        return output;
    }
}

const faceApi = (videoOrOptionsOrCallback, optionsOrCallback, cb) => {
    const {video, options, callback} = new ImageModelArgs(videoOrOptionsOrCallback, optionsOrCallback, cb);
    const instance = new FaceApiBase(video, options, callback);
    return callback ? instance : instance.ready;
};

export default faceApi;
