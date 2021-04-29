// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Sound Classifier using pre-trained networks
*/

import * as tf from '@tensorflow/tfjs';
import * as speechCommands from './speechcommands';
import callCallback, {Callback} from '../utils/callcallback';
import {ArgSeparator} from "../utils/argSeparator";
import {LabelAndConfidence} from "../ImageClassifier";

const MODEL_OPTIONS = ['speechcommands18w'];

/**
 * Need to define the interface of an acceptable model since it can accept multiple types
 */
interface SoundClassifierModel {
  classify(numberOfClasses?: number): Promise<LabelAndConfidence[]>;
}

/**
 * Assume that the options vary from model to model
 */
type SoundClassifierOptions = Record<string, any>;

class SoundClassifier {

  ready: Promise<SoundClassifier>;
  model?: SoundClassifierModel;
  config: SoundClassifierOptions;

  /**
   * Create an SoundClassifier.
   * @param {string} modelNameOrUrl - The name or the URL of the model to use. Current name options
   *    are: 'SpeechCommands18w'.
   * @param {object} options - An object with options.
   * @param {function} callback - A callback to be called when the model is ready.
   */
  constructor(modelNameOrUrl: string, options: SoundClassifierOptions = {}, callback?: Callback<SoundClassifier>) {
    this.config = options;
    // TODO: how should custom URLs be handled?  Right now just loading SpeechCommands with the path as the artifacts URL.  What about metadata?
    const loader = speechCommands.load;
    let url: string | undefined;
    if ( modelNameOrUrl.toLowerCase().endsWith('.json')) {
      url = modelNameOrUrl;
    }
    else if ( modelNameOrUrl.toLowerCase() !== 'speechcommands18w' ) {
      console.warn(`Unknown model name ${modelNameOrUrl}. Reverting to default model "speechcommands18w".`);
    }
    // Load the model
    this.ready = callCallback(this.loadModel(() => loader(options, url)), callback);
  }

  async loadModel(modelLoader: () => Promise<SoundClassifierModel>): Promise<this> {
    this.model = await modelLoader();
    return this;
  }

  async classifyInternal(numberOfClasses?: number, callback?: Callback<LabelAndConfidence[]>): Promise<LabelAndConfidence[]> {
    // Wait for the model to be ready
    await this.ready;
    await tf.nextFrame();

    return callCallback(this.model!.classify(numberOfClasses), callback);
  }

  /**
   * Classifies the audio from microphone and takes a callback to handle the results
   * @param {function | number} numOrCallback -
   *    takes any of the following params
   * @param {function} cb - a callback function that handles the results of the function.
   * @return a promise or the results of a given callback, cb.
   */
  async classify(numOrCallback: number | Callback<LabelAndConfidence[]>, cb?: Callback<LabelAndConfidence[]>): Promise<LabelAndConfidence[]> {
    const {number, callback} = new ArgSeparator(numOrCallback, cb);
    return this.classifyInternal(number, callback);
  }
}

const soundClassifier = (modelName: string, optionsOrCallback?: SoundClassifierOptions | Callback<SoundClassifier>, cb?: Callback<SoundClassifier>) => {
  const {string: model, options, callback} = new ArgSeparator(modelName, optionsOrCallback, cb);
  if (! model) {
    // TODO: should probably just default to speech commands
    throw new Error('Please specify a model to use. E.g: "SpeechCommands18w"');
  }
  const instance = new SoundClassifier(model, options, callback);
  return callback ? instance : instance.ready;
};

export default soundClassifier;
