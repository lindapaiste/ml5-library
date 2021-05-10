// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
/* eslint max-len: ["error", { "code": 180 }] */

/*
YOLO Object detection
Heavily derived from https://github.com/ModelDepot/tfjs-yolo-tiny (ModelDepot: modeldepot.io)
*/

import * as tf from '@tensorflow/tfjs';
import {imgToTensor, TfImageSource,} from "../../utils/imageUtilities";
import CLASS_NAMES from './../../utils/COCO_CLASSES';
import modelLoader from './../../utils/modelLoader';
import {boxesToCorners, filterBoxes, head, nonMaxSuppression,} from './postprocess';
import {ObjectDetector, ObjectDetectorPrediction, specificObjectDetector} from "../index";

export interface YoloOptions {
    modelUrl?: string;
    filterBoxesThreshold?: number;
    IOUThreshold?: number;
    classProbThreshold?: number;
    imageSize?: number;
    disableDeprecationNotice?: boolean;
}

const DEFAULTS: Required<YoloOptions> = {
    modelUrl: 'https://raw.githubusercontent.com/ml5js/ml5-data-and-training/master/models/YOLO/model.json',
    filterBoxesThreshold: 0.01,
    IOUThreshold: 0.4,
    classProbThreshold: 0.4,
    imageSize: 416,
    disableDeprecationNotice: false,
};

export class YOLOBase implements ObjectDetector {

    public config: Required<YoloOptions>;

    private model!: tf.LayersModel;

    // TODO: move to parent ObjectDetector
    public isPredicting: boolean;

    public ready: Promise<ObjectDetector>;

    /**
     * @deprecated Please use ObjectDetector class instead
     */

    /**
     * @typedef {Object} options
     * @property {number} filterBoxesThreshold - default 0.01
     * @property {number} IOUThreshold - default 0.4
     * @property {number} classProbThreshold - default 0.4
     */
    /**
     * Create YOLO model. Works on video and images.
     * @param {Object} options - Optional. A set of options.
     */
    constructor(options: YoloOptions = {}) {
        this.config = {
            ...DEFAULTS,
            ...options
        }
        this.isPredicting = false;
        this.ready = this.loadModel();

        if (!options.disableDeprecationNotice) {
            console.warn("WARNING! Function YOLO has been deprecated, please use the new ObjectDetector function instead");
        }
    }

    async loadModel() {
        // see if the provided URL is to a file or to a directory.
        // if it is a directory, look for the file "model.json"
        const url = modelLoader(this.config.modelUrl).modelJsonPath();
        this.model = await tf.loadLayersModel(url);
        return this;
    }

    /**
     * @typedef {Object} ObjectDetectorPrediction
     * @property {number} x - top left x coordinate of the prediction box in pixels.
     * @property {number} y - top left y coordinate of the prediction box in pixels.
     * @property {number} width - width of the prediction box in pixels.
     * @property {number} height - height of the prediction box in pixels.
     * @property {string} label - the label given.
     * @property {number} confidence - the confidence score (0 to 1).
     * @property {ObjectDetectorPredictionNormalized} normalized - a normalized object of the prediction
     */

    /**
     * @typedef {Object} ObjectDetectorPredictionNormalized
     * @property {number} x - top left x coordinate of the prediction box (0 to 1).
     * @property {number} y - top left y coordinate of the prediction box (0 to 1).
     * @property {number} width - width of the prediction box (0 to 1).
     * @property {number} height - height of the prediction box (0 to 1).
     */
    /**
     * Detect objects that are in video, returns bounding box, label, and confidence scores
     */
    async detect(subject: TfImageSource): Promise<ObjectDetectorPrediction[]> {
        const {imageSize, filterBoxesThreshold, IOUThreshold, classProbThreshold} = this.config;

        const ANCHORS = tf.tensor2d([
            [0.57273, 0.677385],
            [1.87446, 2.06253],
            [3.33843, 5.47434],
            [7.88282, 3.52778],
            [9.77052, 9.16828],
        ]);

        this.isPredicting = true;
        const [allBoxes, boxConfidence, boxClassProbs] = tf.tidy(() => {
            // TODO: what is the purpose of resizing here? Does the model require it?
            const input = imgToTensor(subject, [imageSize, imageSize]);
            const activation = this.model!.predict(input) as tf.Tensor;
            const [boxXY, boxWH, bConfidence, bClassProbs] = head(activation, ANCHORS, 80);
            const aBoxes = boxesToCorners(boxXY, boxWH);
            return [aBoxes, bConfidence, bClassProbs];
        });

        const filtered = await filterBoxes(allBoxes, boxConfidence, boxClassProbs, filterBoxesThreshold);

        allBoxes.dispose();
        boxConfidence.dispose();
        boxClassProbs.dispose();
        // If all boxes have been filtered out
        if (filtered === null) {
            return [];
        }
        const [boxes, scores, classes] = filtered;

        const results: ObjectDetectorPrediction[] = [];

        tf.tidy(() => {
            const width = tf.scalar(imageSize);
            const height = tf.scalar(imageSize);
            const imageDims = tf.stack([height, width, height, width]).reshape([1, 4]);
            const boxesModified = tf.mul(boxes, imageDims);

            const preKeepBoxesArr = boxesModified.dataSync();
            const scoresArr = scores.dataSync();

            const [keepIndx, boxesArr, keepScores] = nonMaxSuppression(
                preKeepBoxesArr,
                scoresArr,
                IOUThreshold,
            );

            const classesIndxArr = classes.gather(tf.tensor1d(keepIndx, 'int32')).dataSync();

            [...classesIndxArr].forEach((classIndx, i) => {
                const classProb = keepScores[i];
                if (classProb < classProbThreshold) {
                    return;
                }

                const className = CLASS_NAMES[classIndx];
                // TODO: Is this really the order? Seems odd.
                let [y, x, h, w] = boxesArr[i];

                y = Math.max(0, y);
                x = Math.max(0, x);
                h = Math.min(imageSize, h) - y;
                w = Math.min(imageSize, w) - x;

                const resultObj = {
                    label: className,
                    confidence: classProb,
                    // TODO: should pixel values be converted back to the original image size?
                    x,
                    y,
                    width: w,
                    height: h,
                    normalized: {
                        x: x / imageSize,
                        y: y / imageSize,
                        width: w / imageSize,
                        height: h / imageSize,
                    }
                };

                results.push(resultObj);
            });

            this.isPredicting = false;

            width.dispose()
            height.dispose()
            imageDims.dispose()
            boxesModified.dispose()
            boxes.dispose();
            scores.dispose();
            classes.dispose();
            ANCHORS.dispose();
        });

        return results;
    }
}

/**
 * export uses ObjectDetector to wrap the YoloBase class and handle callbacks
 */
export const YOLO = specificObjectDetector("yolo");