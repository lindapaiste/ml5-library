// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
/*
 * FaceApi: real-time face recognition, and landmark detection
 * Ported and integrated from all the hard work by: https://github.com/justadudewhohacks/face-api.js?files=1
 */

import * as tf from "@tensorflow/tfjs";
import * as faceapi from "face-api.js";
import callCallback, {Callback} from "../utils/callcallback";
import {ArgSeparator} from "../utils/argSeparator";
import modelLoader from "../utils/modelLoader";
import {TinyYolov2Options} from "face-api.js";

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

interface FaceApiBasicOptions {
    minConfidence?: number;
    withLandmarks?: boolean;
    withDescriptors?: boolean;
    withTinyNet?: boolean;
    withFaceExpressions?: boolean;
    withAgeAndGender?: boolean;
}

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

interface FaceApiModelUrls {
    Mobilenetv1Model: string;
    TinyFaceDetectorModel: string;
    FaceLandmarkModel: string;
    FaceLandmark68TinyNet: string;
    FaceRecognitionModel: string;
}

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

interface HasModelUrls {
    MODEL_URLS: FaceApiModelUrls;
}

interface FaceApiConfig extends FaceApiBasicOptions, HasModelUrls {

}

/**
 * options argument expects the models to be top-level properties.
 *
 * @typedef {FaceApiBasicOptions & FaceApiModelUrls} FaceApiOptions
 */

interface FaceApiOptions extends FaceApiBasicOptions, FaceApiModelUrls {

}

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

// TODO: this is unnecessary
/**
 * Check if the given _param is undefined, otherwise return the _default
 * @param {*} _param
 * @param {*} _default
 */
const checkUndefined = <T>(_param: T | undefined, _default: T): T => {
    return _param !== undefined ? _param : _default;
}


/**
 * @property {HTMLVideoElement | null } video - Video element, if provided
 * @property {faceapi} model - Model from face-api.js package
 * @property {FaceApiConfig} config - Configuration options for the model
 * @property {boolean} modelReady -
 */
class FaceApiBase {
    ready: Promise<FaceApiBase>;
    modelReady: boolean;
    config: Required<FaceApiConfig>;
    model: typeof faceapi;
    video: HTMLVideoElement | null;
    detectorOptions: faceapi.ITinyYolov2Options & faceapi.ISsdMobilenetv1Options;

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
        this.detectorOptions = {};
        this.config = {
            minConfidence: checkUndefined(options.minConfidence, DEFAULTS.minConfidence),
            withLandmarks: checkUndefined(options.withLandmarks, DEFAULTS.withLandmarks),
            withDescriptors: checkUndefined(options.withDescriptors, DEFAULTS.withDescriptors),
            withTinyNet: checkUndefined(options.withTinyNet, DEFAULTS.withTinyNet),
            withAgeAndGender: checkUndefined(options.withAgeAndGender, DEFAULTS.withAgeAndGender),
            withFaceExpressions: checkUndefined(options.withFaceExpressions, DEFAULTS.withFaceExpressions),
            MODEL_URLS: {
                Mobilenetv1Model: checkUndefined(
                    options.Mobilenetv1Model,
                    DEFAULTS.MODEL_URLS.Mobilenetv1Model,
                ),
                TinyFaceDetectorModel: checkUndefined(
                    options.TinyFaceDetectorModel,
                    DEFAULTS.MODEL_URLS.TinyFaceDetectorModel,
                ),
                FaceLandmarkModel: checkUndefined(
                    options.FaceLandmarkModel,
                    DEFAULTS.MODEL_URLS.FaceLandmarkModel,
                ),
                FaceLandmark68TinyNet: checkUndefined(
                    options.FaceLandmark68TinyNet,
                    DEFAULTS.MODEL_URLS.FaceLandmark68TinyNet,
                ),
                FaceRecognitionModel: checkUndefined(
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
                this.config.MODEL_URLS[item] = modelLoader.getModelPath(this.config.MODEL_URLS[item]);
            }
        });

        const {
            Mobilenetv1Model,
            TinyFaceDetectorModel,
            FaceLandmarkModel,
            FaceRecognitionModel,
            FaceLandmark68TinyNet,
        } = this.config.MODEL_URLS;


        if (this.config.withTinyNet) {
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
        if (this.config.withTinyNet) {
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
     * Internal function handles detecting single or multiple faces
     * @param {(HTMLVideoElement | HTMLImageElement | HTMLCanvasElement)} imgToClassify
     * @param {Partial<FaceApiBasicOptions>} faceApiOptions
     * @param {boolean} single
     */
    async detectInternal(imgToClassify, faceApiOptions, single) {
        if ( ! imgToClassify ) {
            // Handle unsupported input
            throw new Error(
                "No input image provided. If you want to classify a video, pass the video element in the constructor. ",
            );
        }
        
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
        result = FaceApiBase.landmarkParts(result);

        return result;
    }

    /**
     * Classifies multiple features by default
     * @param {*} imageOrOptionsOrCallback
     * @param {*} optionsOrCallback
     * @param {*} cb
     */
    async detect(imageOrOptionsOrCallback, optionsOrCallback, cb) {
        const {image, options, callback} = new ArgSeparator(this.video, imageOrOptionsOrCallback, optionsOrCallback, cb);

        return callCallback(this.detectInternal(image, options, false), callback);
    }
    
    /**
     * Classifies a single feature with higher accuracy
     * @param {*} imageOrOptionsOrCallback
     * @param {*} optionsOrCallback
     * @param {*} cb
     */
    async detectSingle(imageOrOptionsOrCallback, optionsOrCallback, cb) {
        const {image, options, callback} = new ArgSeparator(this.video, imageOrOptionsOrCallback, optionsOrCallback, cb);
        return callCallback(this.detectInternal(image, options, true), callback);
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

    /**
     * get parts from landmarks
     * @param {*} result
     */
    static landmarkParts(result: faceapi.FaceDetection | faceapi.FaceDetection[]) {
        let output;
        // multiple detections is an array
        if (Array.isArray(result)) {
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

const faceApi = (videoOrOptionsOrCallback?: HTMLVideoElement | FaceApiOptions | Callback<FaceApiBase>, optionsOrCallback?: FaceApiOptions | Callback<FaceApiBase>, cb?: Callback<FaceApiBase>) => {
    const {video, options, callback} = new ArgSeparator(videoOrOptionsOrCallback, optionsOrCallback, cb);
    const instance = new FaceApiBase(video, options, callback);
    return callback ? instance : instance.ready;
};

export default faceApi;
