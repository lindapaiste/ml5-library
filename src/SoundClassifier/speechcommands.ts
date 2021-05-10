// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tfjsSpeechCommands from '@tensorflow-models/speech-commands';
import {getTopKClassesFromArray, LabelAndConfidence, toLabelAndConfidence} from '../utils/gettopkclasses';
import modelLoader from "../utils/modelLoader";

export class SpeechCommands {
    /**
     * @property model - model created by TensorFlow
     */
    model: tfjsSpeechCommands.SpeechCommandRecognizer;

    /**
     * @property options - configuration options
     */
    options?: tfjsSpeechCommands.StreamingRecognitionConfig;

    /**
     * @property scores - the latest saved result
     */
    private scores?: Float32Array;

    /**
     * @property recognizerPromise - the promise from the inner callback of model.listen
     */
    private recognizerPromise?: Promise<void>;

    /**
     * @param model
     * @param options
     * @constructor
     */
    constructor(model: tfjsSpeechCommands.SpeechCommandRecognizer, options?: tfjsSpeechCommands.StreamingRecognitionConfig) {
        this.model = model;
        this.options = options;
        this.beginListening();
    }

    /**
     * @property allLabels string[]
     */
    get allLabels(): string[] {
        return this.model.wordLabels();
    }

    get isListening(): boolean {
        return this.model.isListening();
    }

    // TODO use EventEmitter
    // Note: the pattern of RecognizerCallback = (result: SpeechCommandRecognizerResult) => Promise<void>; is strange.
    // since it must resolve to void, store the classification as an instance property rather than returning it.
    // TODO: what happens to these rejections?
    private beginListening() {
        this.model.listen(result => {
            this.recognizerPromise = new Promise<void>((resolve, reject) => {
                if (!result.scores) {
                    reject(new Error(`ERROR: Cannot find scores in result: ${result}`));
                }
                const {scores} = result;
                // TODO: result could be 1D or 2D -- how to handle if it is 2D? Flatten? Or reject?
                if (scores instanceof Float32Array) {
                    this.scores = scores;
                    resolve();
                } else {
                    reject(new Error("Unexpected results format. Expected Float32Array but received Float32Array[]"));
                }
            });
            return this.recognizerPromise;
        }, this.options);
    }

    // TODO: I want to make this better, but the pattern of RecognizerCallback = (result: SpeechCommandRecognizerResult) => Promise<void>; is so weird!
    // TODO: what should it await?
    async classify(topk: number = this.allLabels.length): Promise<LabelAndConfidence[]> {
        if ( this.recognizerPromise ) {
            await this.recognizerPromise;
        }
        if (this.scores) {
            return getTopKClassesFromArray(this.scores, topk, this.allLabels)
                .map(toLabelAndConfidence);
        } else {
            throw new Error("Attempted to use the last detection, but no detections were found.");
        }
    }
}

/**
 * @param {string} url - URL is assumed to be the path to the 'model.json' file.
 * It is assumed that a 'metadata.json' file is in the same directory.
 */
const createModel = (url?: string): tfjsSpeechCommands.SpeechCommandRecognizer => {
    if (!url) {
        return tfjsSpeechCommands.create('BROWSER_FFT');
    }
    const loader = modelLoader(url);
    const modelUrl = loader.modelJsonPath();
    const metadataUrl = loader.fileInDirectory("metadata.json");
    return tfjsSpeechCommands.create('BROWSER_FFT', undefined, modelUrl, metadataUrl);
}

/**
 * @param options
 * @param {string} url - can pass a URL to load custom model.json and metadata.json files
 */
export async function load(options?: tfjsSpeechCommands.StreamingRecognitionConfig, url?: string): Promise<SpeechCommands> {
    // model creation is synchronous but the loading is async
    const model = createModel(url);
    await model.ensureModelLoaded();
    return new SpeechCommands(model, options);
}