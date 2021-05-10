// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
/*
* CVAE: Conditional Variational Autoencoder.
* Runs a pro-trained model.
*/

import * as tf from '@tensorflow/tfjs';
import {LayersModel, Rank, Tensor, Tensor4D} from '@tensorflow/tfjs';
import callCallback, {Callback} from '../utils/callcallback';
import {loadFile} from "../utils/io";
import {GeneratedImageResult, generatedImageResult} from "../utils/GeneratedImage";


interface ManifestContents {
    model: string;
    labels: string[];
}

// TODO: make compatible with other generated images
type CVAEResult = Pick<GeneratedImageResult, 'src' | 'image'> & {
    raws: Uint8ClampedArray;
}

/**
 * The fully-loaded, callable model.
 */
class CvaeModel {

    private latentDim: Tensor<Rank.R2>;
    private labelVector: number[];

    constructor(public readonly model: LayersModel, private labels: string[]) {
        this.latentDim = tf.randomUniform([1, 16]);
        // get an array full of zero with the length of labels [0, 0, 0 ...]
        // TODO: why is there a +1?
        this.labelVector = Array(this.labels.length + 1).fill(0);
    }

    /**
     * Generate a random image for a given label.
     * @param label - A label of the feature your want to generate
     */
    public async generate(label: string): Promise<CVAEResult> {
        const cursor = this.labels.indexOf(label);
        if (cursor < 0) {
            throw new Error(`Label "${label}" is not one of the valid labels for this model`);
        }

        const res = tf.tidy(() => {
            this.latentDim = tf.randomUniform([1, 16]);

            this.labelVector = this.labelVector.map(() => 0); // clear vector
            this.labelVector[cursor + 1] = 1;

            const input = tf.tensor<Rank.R1>([this.labelVector]);

            // type assertion needed because it could return one tensor or an array or tensors
            // TODO: are ranks correct? seems like an R2 and an R1...
            const temp = this.model.predict([this.latentDim, input]) as Tensor4D;
            return temp.reshape<Rank.R3>([temp.shape[1], temp.shape[2], temp.shape[3]]);
        });

        const result = await generatedImageResult(res, {returnTensors: true});

        return {
            ...result,
            raws: result.raw.data
        }
        // TODO: fix "raw" / "raws" inconsistency
    }
}

/**
 * Asynchronous function to create the model instance.
 */
const createCvaeModel = async (modelPath: string): Promise<CvaeModel> => {
    const manifest = await loadFile<ManifestContents>(modelPath);
    const modelPathPrefix = modelPath.split('manifest.json')[0];
    const modelUrl = modelPathPrefix + manifest.model;
    const model = await tf.loadLayersModel(modelUrl);
    return new CvaeModel(model, manifest.labels);
}

/**
 * For compatibility with existing setup.
 * Combine the model and the create function into a class which is compatible with the previous class
 * and can be used interchangeably.
 */
class Cvae {
  public ready: Promise<any>; // is specified in docs as a boolean
  public model?: LayersModel; // not actually documented so maybe unneeded
  private cvae?: CvaeModel; // the underlying model class

  constructor(modelPath: string, callback?: Callback<Cvae>) {
    this.ready = callCallback( (async () => {
      this.cvae = await createCvaeModel(modelPath);
      this.model = this.cvae.model;
      return this;
    })(), callback)
  }

  async generate(label: string, callback?: Callback<CVAEResult>): Promise<CVAEResult> {
    await this.ready;
    return callCallback(this.cvae!.generate(label), callback);
  }
}

/**
 * @param {string} model - A path to a manifest.json file.  That manifest is expected to contain:
 *  - a property `model` with the path of the model, relative to the folder containing the manifest.json
 *  - a property `labels` which is an array of strings
 * @param {function} callback
 * @constructor
 */
const CVAE = (model: string, callback?: Callback<Cvae>) => new Cvae(model, callback);


export default CVAE;
