// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* eslint prefer-destructuring: ["error", {AssignmentExpression: {array: false}}] */
/* eslint no-await-in-loop: "off" */
/*
A LSTM Generator: Run inference mode for a pre-trained LSTM.
*/

import * as tf from "@tensorflow/tfjs";
import {Rank, Tensor, Tensor1D} from "@tensorflow/tfjs";
import sampleFromDistribution from "./../utils/sample";
import CheckpointLoader from "../utils/checkpointLoader";
import callCallback, {Callback} from "../utils/callcallback";
import {Tensor2D} from "@tensorflow/tfjs-core";
import {isRank} from "../utils/imageConversion";
import {loadFile} from "../utils/io";
import {TensorLike} from "@tensorflow/tfjs-core/src/types";

interface State {
  c: Tensor2D[];
  h: Tensor2D[];
}

const ZERO_STATE = { c: [], h: [] };

/**
 * @typedef {Object} options
 * @property {String} seed
 * @property {number} length
 * @property {number} temperature
 */

export interface CharRNNOptions {
  seed: string;
  length: number;
  temperature: number;
  stateful: boolean;
}

const DEFAULTS = {
  seed: "a", // TODO: use no seed by default
  length: 20,
  temperature: 0.5,
  stateful: false,
};

export interface GenerateResult {

}

// non-capture group matches either "cell" ot "lstm"
// captures the number
const regexCell = new RegExp(/(?:cell|lstm)_([0-9])/gi);
// matches "weight" or "kernel"
const regexWeights = new RegExp(/(weight|kernel)/gi);
// matches "softmax_w" and "softmax_b"
// captures the letter "w" or "b"
const regexFullyConnected = new RegExp(/softmax_([wb])/gi);

interface LoadedVars {
  kernelArray: Tensor2D[];
  biasArray: Tensor1D[];
  fullyConnectedWeights: Tensor;
  fullyConnectedBiases: Tensor;
  embedding?: Tensor2D;
}

const processVars = (vars: Record<string, Tensor>): LoadedVars => {
  const kernelArray: Tensor2D[] = [];
  const biasArray: Tensor1D[] = [];
  let fullyConnectedWeights;
  let fullyConnectedBiases;
  let embedding;

  Object.entries(vars).forEach(([key, tensor]) => {
    // Note: if bias/kernel always comes after cell_# then this could be cleaned up to one regex
    let match = regexCell.exec(key);
    if (match) {
      const index = parseInt(match[1]); // index from capture group
      // is weights
      if (regexWeights.test(key)) {
        if (! isRank(tensor, Rank.R2)) {
          throw new Error(`Cell kernel tensor must be rank ${Rank.R2}. Variable ${key} has rank ${tensor.rankType}`);
        }
        kernelArray[index] = tensor;
      }
      // is biases
      else {
        if (! isRank(tensor, Rank.R1)) {
          throw new Error(`Cell bias tensor must be rank ${Rank.R1}. Variable ${key} has rank ${tensor.rankType}`);
        }
        biasArray[index] = tensor;
      }
    }
    match = regexFullyConnected.exec(key);
    if (match) {
      if (match[1] === "w") {
        fullyConnectedWeights = tensor;
      } else {
        fullyConnectedBiases = tensor;
      }
    }
    if ( key.match(/embedding/gi)) {
      embedding = tensor;
    }
  });

}


class CharRNNModel {
  kernelArray: Tensor2D[] = [];
  biasArray: Tensor1D[] = [];
  fullyConnectedWeights: Tensor;
  fullyConnectedBiases: Tensor;
  embedding: Tensor;

  constructor(vars: Record<string, Tensor>) {
    // need to make sure that these are always set
    let fullyConnectedWeights;
    let fullyConnectedBiases;
    Object.entries(vars).forEach(([key, tensor]) => {
      // Note: if bias/kernel always comes after cell_# then this could be cleaned up to one regex
      let match = regexCell.exec(key);
      if (match) {
        const index = parseInt(match[1]); // index from capture group
        if (regexWeights.test(key)) {
          this.kernelArray[index] = tensor;
          return;
        } else {
          this.biasArray[index] = tensor;
          return;
        }
      }
      match = regexFullyConnected.exec(key);
      if (match) {
        if (match[1] === "w") {
          this.fullyConnectedWeights = tensor;
        } else {
          this.fullyConnectedBiases = tensor;
        }
      }
    });
  }

}

type Cell = (data: Tensor2D|TensorLike, c: Tensor2D|TensorLike, h: Tensor2D|TensorLike) => [Tensor2D, Tensor2D];


class CharRNN {
  ready: Promise<this>;
  cellsAmount: number;
  cells: Cell[];
  state: State;
  zeroState: State;
  /**
   * Mapping of each distinct character to a unique number
   */
  vocab: Record<string, number>;
  /**
   * The vocabulary size (or total number of possible characters).
   */
  vocabSize: number;
  probabilities: number[];
  defaults: CharRNNOptions; //TODO: this doesn't belong on the instance
  /**
   * The pre-trained charRNN model.
   */
  model: LoadedVars;
  /**
   * Create a CharRNN.
   * @param {String} modelPath - The path to the trained charRNN model.
   * @param {function} callback  - Optional. A callback to be called once
   *    the model has loaded. If no callback is provided, it will return a
   *    promise that will be resolved once the model has loaded.
   */
  constructor(modelPath: string, callback?: Callback<CharRNN>) {
    /**
     * Boolean value that specifies if the model has loaded.
     * @type {boolean}
     * @public
     */
    //this.ready = false;

    this.model = {};
    this.cellsAmount = 0;
    this.cells = [];
    this.zeroState = { c: [], h: [] };
    this.state = { c: [], h: [] };
    this.vocab = {};
    this.vocabSize = 0;
    this.probabilities = [];
    this.defaults = {
      seed: "a", // TODO: use no seed by default
      length: 20,
      temperature: 0.5,
      stateful: false,
    };

    this.ready = callCallback(this.loadCheckpoints(modelPath), callback);
  }

  setState(state: State) {
    this.state = state;
  }

  getState() {
    return this.state;
  }

  private charToNum = (char: string): number => {
    return this.vocab[char];
  }

  private numToChar = (num: number): string => {
    const [char] = Object.entries(this.vocab).find(([_, n]) => n === num)!;
    return char;
  }

  async loadCheckpoints(path: string) {
    const reader = new CheckpointLoader(path);
    const vars = await reader.getAllVariables();
    Object.keys(vars).forEach(key => {
      if (key.match(regexCell)) {
        if (key.match(regexWeights)) {
          this.model[`Kernel_${key.match(/[0-9]/)[0]}`] = vars[key];
          this.cellsAmount += 1;
        } else {
          this.model[`Bias_${key.match(/[0-9]/)[0]}`] = vars[key];
        }
      } else if (key.match(regexFullyConnected)) {
        if (key.match(regexWeights)) {
          this.model.fullyConnectedWeights = vars[key];
        } else {
          this.model.fullyConnectedBiases = vars[key];
        }
      } else {
        this.model[key] = vars[key];
      }
    });
    await this.loadVocab(path);
    await this.initCells();
    return this;
  }

  async loadVocab(path: string) {
      this.vocab = await loadFile(`${path}/vocab.json`);
      this.vocabSize = Object.keys(this.vocab).length;
      return this.vocab;
  }

  /**
   * Creates the cells.
   * Creates the zeroState and sets this.state to it.
   */
  async initCells() {
    this.cells = [];
    this.zeroState = { c: [], h: [] };
    const forgetBias = tf.tensor<Rank.R0>(1.0);

    const lstm = (i: number): Cell => {
      return (DATA, C, H) =>
          tf.basicLSTMCell(
              forgetBias,
              this.model[`Kernel_${i}`],
              this.model[`Bias_${i}`],
              DATA,
              C,
              H,
          );
    };

    for (let i = 0; i < this.cellsAmount; i += 1) {
      this.zeroState.c.push(tf.zeros([1, this.model[`Bias_${i}`].shape[0] / 4]));
      this.zeroState.h.push(tf.zeros([1, this.model[`Bias_${i}`].shape[0] / 4]));
      this.cells.push(lstm(i));
    }

    this.state = this.zeroState;
  }

  // duplicated code from inside the loops of feed and generateInternal
  private async iteration(input: number) {
    const onehotBuffer = await tf.buffer<tf.Rank.R2>([1, this.vocabSize]);
    onehotBuffer.set(1.0, 0, input);
    const onehot = onehotBuffer.toTensor();
    let output;
    if (this.model.embedding) {
      const embedded = tf.matMul(onehot, this.model.embedding);
      output = tf.multiRNNCell(this.cells, embedded, this.state.c, this.state.h);
    } else {
      output = tf.multiRNNCell(this.cells, onehot, this.state.c, this.state.h);
    }

    this.state.c = output[0];
    this.state.h = output[1];

    // TODO: dispose
  }

  async generateInternal(options: CharRNNOptions) {
    await this.ready;
    const seed = options.seed || this.defaults.seed;
    const length = +options.length || this.defaults.length;
    const temperature = +options.temperature || this.defaults.temperature;
    const stateful = options.stateful || this.defaults.stateful;
    if (!stateful) {
      this.state = this.zeroState;
    }

    const results = [];
    const userInput = Array.from(seed);
    const encodedInput = userInput.map(this.charToNum);

    let input = encodedInput[0];
    let probabilitiesNormalized; // will contain final probabilities (normalized)

    for (let i = 0; i < userInput.length + length + -1; i += 1) {
      await this.iteration(input);

      probabilitiesNormalized = this.calcProbabilitiesNormalized(temperature);

      if (i < userInput.length - 1) {
        input = encodedInput[i + 1];
      } else {
        input = sampleFromDistribution(probabilitiesNormalized);
        results.push(input);
      }
    }

    const generated = results.map(this.numToChar).join();

    this.probabilities = probabilitiesNormalized || [];
    return {
      sample: generated,
      state: this.state,
    };
  }

  // from duplicated code block
  calcProbabilitiesNormalized(temperature: number): number[] {
    return tf.tidy(() => {
      const outputH = this.state.h[1];
      const weightedResult = tf.matMul(outputH, this.model.fullyConnectedWeights);
      const logits = tf.add(weightedResult, this.model.fullyConnectedBiases);
      const divided = tf.div(logits, temperature);
      const probabilities = tf.exp(divided);
      const probabilitiesNormalized = tf.div(probabilities, tf.sum(probabilities)).dataSync();
      return [...probabilitiesNormalized];
    });
  }

  /**
   * Reset the model state.
   */
  reset() {
    this.state = ZERO_STATE;
  }



  // stateless
  /**
   * Generates content in a stateless manner, based on some initial text
   *    (known as a "seed"). Returns a string.
   * @param {options} options - An object specifying the input parameters of
   *    seed, length and temperature. Default length is 20, temperature is 0.5
   *    and seed is a random character from the model. The object should look like
   *    this:
   * @param {function} callback - Optional. A function to be called when the model
   *    has generated content. If no callback is provided, it will return a promise
   *    that will be resolved once the model has generated new content.
   */
  async generate(options: CharRNNOptions, callback) {
    this.reset();
    return callCallback(this.generateInternal(options), callback);
  }

  // stateful
  /**
   * Predict the next character based on the model's current state.
   * @param {number} temp
   * @param {function} callback - Optional. A function to be called when the
   *    model finished adding the seed. If no callback is provided, it will
   *    return a promise that will be resolved once the prediction has been generated.
   */
  async predict(temp: number, callback) {
    const temperature = temp > 0 ? temp : 0.1;
    const probabilitiesNormalized = this.calcProbabilitiesNormalized(temperature);

    const sample = sampleFromDistribution(probabilitiesNormalized);
    const nextChar = this.numToChar(sample);
    this.probabilities = probabilitiesNormalized;
    if (callback) {
      callback(nextChar);
    }
    const pm = Object.keys(this.vocab).map(c => ({
      char: c,
      probability: this.probabilities[this.vocab[c]],
    }));
    return {
      sample: nextChar,
      probabilities: pm,
    };
  }

  /**
   * Feed a string of characters to the model state.
   * @param {String} inputSeed - A string to feed the charRNN model state.
   * @param {function} callback  - Optional. A function to be called when
   *    the model finished adding the seed. If no callback is provided, it
   *    will return a promise that will be resolved once seed has been fed.
   */
  async feed(inputSeed: string, callback) {
    await this.ready;
    const seed = Array.from(inputSeed);
    const encodedInput = seed.map(this.charToNum);
    for (const input of encodedInput) {
      await this.iteration(input);
    }

    if (callback) {
      callback();
    }
  }
}

const charRNN = (modelPath = "./", callback) => new CharRNN(modelPath, callback);

export default charRNN;
