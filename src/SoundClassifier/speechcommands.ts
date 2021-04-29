// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tfjsSpeechCommands from '@tensorflow-models/speech-commands';
import {getTopKClassesFromArray} from '../utils/gettopkclasses';
import {LabelAndConfidence} from "../ImageClassifier";

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
   * @param model
   * @param options
   * @constructor
   */
  constructor(model: tfjsSpeechCommands.SpeechCommandRecognizer, options?: tfjsSpeechCommands.StreamingRecognitionConfig) {
    this.model = model;
    this.options = options;
  }

  /**
   * @property allLabels string[]
   */
  get allLabels(): string[] {
    return this.model.wordLabels();
  }

  private _handleResult(result: tfjsSpeechCommands.SpeechCommandRecognizerResult, topk: number): LabelAndConfidence[] {
    // TODO: this seems unnecessary.  TS says results is always set.
    if (!result.scores) {
      throw new Error(`ERROR: Cannot find scores in result: ${result}`)
    }
    const {scores} = result;
    // TODO: result could be 1D or 2D -- how to handle if it is 2D?
    if (scores instanceof Float32Array) {
      return getTopKClassesFromArray(scores, topk, this.allLabels)
          .map(c => ({label: c.className, confidence: c.probability}));
    } else {
       throw new Error("Unexpected results format. Expected Float32Array but received Float32Array[]");
    }
  }

  // TODO: I want to make this better, but the pattern of RecognizerCallback = (result: SpeechCommandRecognizerResult) => Promise<void>; is so weird!
  // right now it only works with a callback, does not return a promise that resolves to the result
  classify(topk: number = this.allLabels.length): Promise<LabelAndConfidence[]> {
    return new Promise((resolve, reject) => {
      this.model.listen(result => {
        if (!result.scores) {
          reject(new Error(`ERROR: Cannot find scores in result: ${result}`) );
        }
        const {scores} = result;
        // TODO: result could be 1D or 2D -- how to handle if it is 2D?
        if (scores instanceof Float32Array) {
          const classes = getTopKClassesFromArray(scores, topk, this.allLabels)
              .map(c => ({label: c.className, confidence: c.probability}));
          resolve(classes);
        } else {
          reject(new Error("Unexpected results format. Expected Float32Array but received Float32Array[]") );
        }
      }, this.options).catch(reject);
    });
  }
}

const createModel = (url?: string): tfjsSpeechCommands.SpeechCommandRecognizer => {
  if (url) {
    const split = url.split("/");
    const prefix = split.slice(0, split.length - 1).join("/");
    const metadataJson = `${prefix}/metadata.json`;
    return tfjsSpeechCommands.create('BROWSER_FFT', undefined, url, metadataJson);
  } else {
    return tfjsSpeechCommands.create('BROWSER_FFT');
  }
}

/**
 * @param options
 * @param {string} url can pass a URL to load a custom metadata.json file
 */
export async function load(options?: tfjsSpeechCommands.StreamingRecognitionConfig, url?: string): Promise<SpeechCommands> {
  // model creation is synchronous but the loading is async
  const model = createModel(url);
  await model.ensureModelLoaded();
  return  new SpeechCommands(model, options);
}