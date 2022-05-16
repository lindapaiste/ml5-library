// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
SketchRNN
*/

import * as ms from '@magenta/sketch';
import callCallback, {ML5Callback} from '../utils/callcallback';
import modelPaths from './models';
import {getModelPath} from '../utils/modelLoader';
import {LSTMState} from "@magenta/sketch/es5/sketch_rnn/model";
import {ArgSeparator} from "../utils/argSeparator";

/**
 * Combine a model name like 'cat' and a size into a the absolute URL for the model on google cloud storage
 *
 * @param {string} model
 * @param {boolean} large
 * @return {string}
 */
const createPath = (model: string, large: boolean) => {
    return `https://storage.googleapis.com/quickdraw-models/sketchRNN/${large ? "large_models" : "models"}/${model}.gen.json`
}

/**
 * @typedef {Object} SketchRNNOptions
 * @property {number} temperature - default 0.65
 * @property {number} pixelFactor - default 3.0
 * @property {boolean} large - default true
 */
interface SketchRNNOptions {
    temperature?: number;
    pixelFactor?: number;
    large?: boolean;
}

const DEFAULTS = {
    temperature: 0.65,
    pixelFactor: 3.0,
    large: true,
}

/**
 * @typedef {Object} PenStroke
 * @property {number} dx - The x location of the pen.
 * @property {number} dy - The y location of the pen.
 * @property {string} pen - One of: 'up', 'down', 'end'.
 *    Determines whether the pen is down, up, or if the stroke has ended.
 */
interface PenStroke {
    dx: number;
    dy: number;
    pen: 'up' | 'down' | 'end';
}


/**
 * @property {SketchRNNOptions} config
 * @property {ms.SketchRNN} model - magenta sketch model
 * @property {Promise} ready
 */
class SketchRNN {
    config: Required<SketchRNNOptions>;
    model: ms.SketchRNN;
    penState: number[];
    ready: Promise<this>;
    rnnState?: LSTMState;

    /**
     * Create SketchRNN.
     * @param {String} model - The name of the sketch model to be loaded.
     *    The names can be found in the models.js file
     * @param {SketchRNNOptions} options
     * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise
     *    that will be resolved once the model has loaded.
     */
    constructor(model: string, options?: SketchRNNOptions, callback?: ML5Callback<SketchRNN>) {
        this.config = {
            ...DEFAULTS,
            ...options
        };
        // see if the model is an accepted name or a URL
        const modelUrl = modelPaths.has(model) ? createPath(model, this.config.large) : getModelPath(model);
        // create the model
        this.model = new ms.SketchRNN(modelUrl);
        this.penState = this.model.zeroInput();
        this.ready = callCallback(this.model.initialize().then(() => this), callback);
    }

    /**
     * @param {SketchRNNOptions} options
     * @param {number[][]} strokes
     * @return {Promise<{PenMovement}>}
     */
    async generateInternal(options: SketchRNNOptions, strokes: number[][]): Promise<PenStroke> {
        const temperature = +(options.temperature || this.config.temperature);
        const pixelFactor = +(options.pixelFactor || this.config.pixelFactor);

        await this.ready;
        if (!this.rnnState) {
            this.rnnState = this.model.zeroState();
            this.model.setPixelFactor(pixelFactor);
        }

        if (Array.isArray(strokes) && strokes.length) {
            this.rnnState = this.model.updateStrokes(strokes, this.rnnState);
        }
        this.rnnState = this.model.update(this.penState, this.rnnState);
        const pdf = this.model.getPDF(this.rnnState, temperature);
        this.penState = this.model.sample(pdf);

        const [dx, dy, isDown, isUp, isEnd] = this.penState;

        return {
            dx,
            dy,
            pen: isDown ? 'down' : isUp ? 'up' : 'end'
        };
    }

    /**
     * @param {PenStroke[] | SketchRNNOptions | function} optionsOrSeedOrCallback
     * @param {SketchRNNOptions | function} [optionsOrCallback]
     * @param {function} [cb]
     * @return {Promise<PenStroke>}
     */
    async generate(
        optionsOrSeedOrCallback: PenStroke[] | SketchRNNOptions | ML5Callback<PenStroke>,
        optionsOrCallback: SketchRNNOptions | ML5Callback<PenStroke>,
        cb: ML5Callback<PenStroke>,
    ): Promise<PenStroke> {
        const {array: seedStrokes = [], options = {}, callback} = new ArgSeparator(optionsOrSeedOrCallback, optionsOrCallback, cb);

        const strokes = seedStrokes.map(s => {
            const up = s.pen === 'up' ? 1 : 0;
            const down = s.pen === 'down' ? 1 : 0;
            const end = s.pen === 'end' ? 1 : 0;
            return [s.dx, s.dy, down, up, end];
        });

        return callCallback(this.generateInternal(options, strokes), callback);
    }

    /**
     * reset the model to its original state
     */
    reset() {
        this.penState = this.model.zeroInput();
        if (this.rnnState) {
            this.rnnState = this.model.zeroState();
        }
    }
}

/**
 * @param {string} model - Required. The name of the sketch model to be loaded.
 *    The names can be found in the models.js file.
 *    Can also provide a path to a model file in '.gen.json' format.
 * @param {function} [callback] - Optional. A callback function that is called once the model has loaded.
 *    If no callback is provided, it will return a promise that will be resolved once the model has loaded.
 * @param {boolean} [large] // TODO: accept options object - this is a breaking change so I need to check that it's ok
 * @return {SketchRNN}
 */
const sketchRNN = (model: string, callback: ML5Callback<SketchRNN>, large = true) => new SketchRNN(model, {large}, callback);

export default sketchRNN;
