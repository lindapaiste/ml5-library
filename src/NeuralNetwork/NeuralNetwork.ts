import * as tf from '@tensorflow/tfjs';
import axios from 'axios';
import callCallback, {Callback} from '../utils/callcallback';
import {saveBlob} from '../utils/io';
import {randomGaussian} from '../utils/random';
import {Tensor} from "@tensorflow/tfjs-core";
import {ArrayMap} from "@tensorflow/tfjs-core/dist/types";
import {ArgSeparator} from "../utils/argSeparator";

/**
 * Class for interacting with the model itself, which is a TensorFlow layers model.
 */

// note: used arrow function to replace constructor binding, but unsure if this is even necessary

export interface TrainingOptions extends tf.ModelFitArgs {
    // note: tf model can also accept an array or keyed object of Tensors,
    // but current code calls .dispose() so it expects only one Tensor.
    inputs: Tensor;
    outputs: Tensor;
    whileTraining?: tf.ModelFitArgs['callbacks'];
}

class NeuralNetwork {
    isTrained: boolean;
    isCompiled: boolean;
    isLayered: boolean;

    model: tf.Sequential;

    constructor() {
        // Initialize flags with default value of false
        this.isTrained = false;
        this.isCompiled = false;
        this.isLayered = false;

        // Model creation is synchronous, so can assign it directly in the constructor
        this.model = this.createModel();
    }

    /**
     * creates a sequential model
     * uses switch/case for potential future where different formats are supported
     * @param {*} type
     */
    createModel = (type = 'sequential'): tf.Sequential => {
        switch (type.toLowerCase()) {
            case 'sequential':
            default:
                return tf.sequential();
        }
    }

    /**
     * add layer to the model
     * if the model has 2 or more layers switch the isLayered flag
     * @param {*} layerOptions
     */
    addLayer = (layerOptions: tf.layers.Layer) => {
        this.model.add(layerOptions);

        // check if it has at least an input and output layer
        if (this.model.layers.length >= 2) {
            this.isLayered = true;
        }
    }

    /**
     * Compile the model
     * if the model is compiled, set the isCompiled flag to true
     * @param {*} modelOptions
     */
    compile = (modelOptions: tf.ModelCompileArgs): void => {
        this.model.compile(modelOptions);
        this.isCompiled = true;
    }

    /**
     * Set the optimizer function given the learning rate
     * as a parameter
     * @param {*} learningRate
     * @param {*} optimizer
     */
    setOptimizerFunction = <T>(learningRate: number, optimizer: (this: this, rate: number) => T): T => {
        return optimizer.call(this, learningRate);
    }

    /**
     * Calls the trainInternal() and calls the callback when finished
     * @param {*} options
     * @param {*} _cb
     */
    train = (options: TrainingOptions, _cb?: Callback<void>): Promise<void> => {
        return callCallback(this.trainInternal(options), _cb);
    }

    /**
     * Train the model
     * @param {*} options
     */
    trainInternal = async (options: TrainingOptions): Promise<void> => {
        // config contains: batchSize, epochs, shuffle, validationSplit
        const {inputs: xs, outputs: ys, whileTraining, ...config} = options;

        await this.model.fit(xs, ys, {
            ...config,
            callbacks: whileTraining,
        });

        xs.dispose();
        ys.dispose();

        this.isTrained = true;
    }

    /**
     * returns the prediction as an array synchronously
     * @param {*} inputs
     */
    predictSync = (inputs: tf.Tensor): ArrayMap[tf.Rank] => {
        const output = tf.tidy(() => {
            return this.model.predict(inputs) as tf.Tensor;
        });
        const result = output.arraySync();

        output.dispose();
        inputs.dispose();

        return result;
    }

    /**
     * returns the prediction as an array
     * @param {*} inputs
     */
    predict = async (inputs: tf.Tensor): Promise<ArrayMap[tf.Rank]> => {
        const output = tf.tidy(() => {
            // TODO: what about options batchSize and verbose?
            return this.model.predict(inputs) as tf.Tensor;
        });
        const result = await output.array();

        output.dispose();
        inputs.dispose();

        return result;
    }

    /**
     * classify is the same as .predict()
     * @param {*} _inputs
     */
    classify = this.predict;

    /**
     * classify is the same as .predict()
     * @param {*} _inputs
     */
    classifySync = this.predictSync;

    // predictMultiple
    // classifyMultiple
    // are the same as .predict()

    /**
     * save the model
     * @param {*} nameOrCb
     * @param {*} cb
     */
    save = async (nameOrCb?: string | Callback<any>, cb?: Callback<tf.io.SaveResult>): Promise<tf.io.SaveResult> => {
        const {string: modelName = 'model', callback} = new ArgSeparator(nameOrCb, cb);
        return callCallback(this.model.save(
            tf.io.withSaveHandler(async data => {
                const weightsManifest = {
                    modelTopology: data.modelTopology,
                    weightsManifest: [
                        {
                            paths: [`./${modelName}.weights.bin`],
                            weights: data.weightSpecs,
                        },
                    ],
                };

                // it is possible that data.weightData is undefined.  Should this be an error?
                await saveBlob(data.weightData || '', `${modelName}.weights.bin`, 'application/octet-stream');
                await saveBlob(JSON.stringify(weightsManifest), `${modelName}.json`, 'text/plain');
                return {
                    modelArtifactsInfo: {
                        dateSaved: new Date(),
                        modelTopologyType: "JSON",
                    }
                    // can also include responses and errors
                }
            })
        ), callback);
    }

    /**
     * loads the model and weights
     * @param {*} filesOrPath
     * @param {*} callback
     */
        // TODO: clean this up
    load = async (filesOrPath: string | FileList | { model: string; weights: string }, callback?: Callback<tf.Sequential>): Promise<tf.Sequential> => {
        if (filesOrPath instanceof FileList) {
            const files = await Promise.all(
                Array.from(filesOrPath).map(async file => {
                    if (file.name.includes('.json') && !file.name.includes('_meta')) {
                        return {name: 'model', file};
                    } else if (file.name.includes('.json') && file.name.includes('_meta.json')) {
                        const modelMetadata = await file.text();
                        return {name: 'metadata', file: modelMetadata};
                    } else if (file.name.includes('.bin')) {
                        return {name: 'weights', file};
                    }
                    return {name: null, file: null};
                }),
            );

            const model = files.find(item => item.name === 'model')?.file;
            const weights = files.find(item => item.name === 'weights')?.file;

            // load the model
            this.model = await tf.loadLayersModel(tf.io.browserFiles([model, weights])) as tf.Sequential;
        } else if (typeof filesOrPath === "object") {
            // load the modelJson
            const modelJsonResult = await axios.get(filesOrPath.model, {responseType: 'text'});
            const modelJson = JSON.stringify(modelJsonResult.data);
            // TODO: browser File() API won't be available in node env
            const modelJsonFile = new File([modelJson], 'model.json', {type: 'application/json'});

            // load the weights
            const weightsBlobResult = await axios.get(filesOrPath.weights, {responseType: 'blob'});
            const weightsBlob = weightsBlobResult.data;
            // TODO: browser File() API won't be available in node env
            const weightsBlobFile = new File([weightsBlob], 'model.weights.bin', {
                type: 'application/macbinary',
            });

            // TODO: does it actually create a model of the Sequential type?
            this.model = await tf.loadLayersModel(tf.io.browserFiles([modelJsonFile, weightsBlobFile])) as tf.Sequential;
        } else {
            this.model = await tf.loadLayersModel(filesOrPath) as tf.Sequential;
        }

        this.isCompiled = true;
        this.isLayered = true;
        this.isTrained = true;

        // TODO: proper callback handling
        if (callback) {
            callback();
        }
        // TODO: why return this.model instead of this?
        return this.model;
    }

    /**
     * dispose and release the memory for the model
     */
    dispose = () => {
        this.model.dispose();
    }

    /**
     * Create a copy of the model with the same layers and weights.
     */
    clone = () => {
        this.model.getUserDefinedMetadata()
    }

    // NeuroEvolution Functions

    /**
     * mutate the weights of a model
     * @param {*} rate
     * @param {*} mutateFunction
     */
    mutate = (rate: number = 0.1, mutateFunction?: (value: number) => number): void => {
        tf.tidy(() => {
            const weights = this.model.getWeights();
            // use randomGaussian as the default mutate function
            const random = randomGaussian(0, 1);
            const mutatedWeights = weights.map(tensor => {
                // create a new tensor where some values are mutated and others are not
                const {shape} = tensor;
                // TODO: Evaluate if this should be sync or not
                const values = tensor.dataSync().slice();
                for (let j = 0; j < values.length; j += 1) {
                    if (Math.random() < rate) {
                        if (mutateFunction) {
                            values[j] = mutateFunction(values[j]);
                        } else {
                            // TODO: is it really necessary or desired to clamp weight values between -1 and 1?
                            // Note: resample if out of bounds is better than clamp, but I think this is not needed at all
                            values[j] = Math.min(Math.max(values[j] + random(), -1), 1);
                        }
                    }
                }
                return tf.tensor(values, shape);
            });
            this.model.setWeights(mutatedWeights);
        });
    }

    /**
     * create a new neural network with crossover
     * TODO: it actually merges into the current -- should it create a new instance instead?
     * @param {*} other
     */
    crossover = (other: NeuralNetwork): void => {
        return tf.tidy(() => {
            const weightsA = this.model.getWeights();
            const weightsB = other.model.getWeights();
            const childWeights = [];
            for (let i = 0; i < weightsA.length; i += 1) {
                const tensorA = weightsA[i];
                const tensorB = weightsB[i];
                const {shape} = weightsA[i];
                // TODO: Evaluate if this should be sync or not
                const valuesA = tensorA.dataSync().slice();
                const valuesB = tensorB.dataSync().slice();
                for (let j = 0; j < valuesA.length; j += 1) {
                    if (Math.random() < 0.5) {
                        valuesA[j] = valuesB[j];
                    }
                }
                childWeights[i] = tf.tensor(valuesA, shape);
            }
            this.model.setWeights(childWeights);
        });
    }
}

export default NeuralNetwork;