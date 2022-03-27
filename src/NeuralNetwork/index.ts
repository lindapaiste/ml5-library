import * as tf from '@tensorflow/tfjs';
import NeuralNetwork, {TrainingOptions} from './NeuralNetwork';
import NeuralNetworkData, {KeyedInputMeta, Metadata, NormalizedData, RawData} from './NeuralNetworkData';
import NeuralNetworkVis from './NeuralNetworkVis';
import callCallback, {Callback} from '../utils/callcallback';

import nnUtils from './NeuralNetworkUtils';
import {toPixels} from "../utils/imageConversion";
import {ArgSeparator} from "../utils/argSeparator";
import findTask, {NNTask, TASKS} from "./tasks";
import {LossOrMetricFn} from "@tensorflow/tfjs-layers/src/types";
import {LabelAndConfidence} from "../utils/gettopkclasses";

// match the arguments for a layer creation function with its arguments
type TFLayers = typeof tf.layers;
export type LayerJson = {
    [K in keyof TFLayers]: TFLayers[K] extends (args: infer A) => tf.layers.Layer ? {
        type: K
    } & A : never;
}[keyof TFLayers]

// it's ok for units to be missing on last layer because it will be filled in with the output type
export type LayerJsonArray = [...LayerJson[], Omit<LayerJson, 'units'>]

export type Prediction = {
    label: string;
    value: number;
    unNormalizedValue?: number;
}

/*If an array is given, then the input values should match the order that the data are specified in the inputs of the constructor options.
If an object is given, then the input values should be given as a key/value pair. The keys must match the keys given in the inputs of the constructor options and/or the keys added when the data were added in .addData().*/
export type Input = Record<string, string | number> | (string | number)[];

export const DEFAULTS = {
    inputs: [],
    outputs: [],
    dataUrl: null,
    modelUrl: null,
    layers: [],
    //task: null,
    debug: false,
    learningRate: 0.2,
    hiddenUnits: 16,
    noTraining: false,
};

export type Label = string | number;

export interface NeuralNetworkOptions {
    /**
     * Required - Can be:
     * number - count of input properties
     * number[] - image shape as [width, height, channels]
     * string[] - names of the properties to use as inputs
     */
    inputs: number | string[] | number[],
    /**
     * Required - Can be:
     * number - count of output properties
     * string[] - names of the properties to use as outputs
     * TODO: is this ever number[]?
     */
    outputs: number | string[],
    /**
     * Optional - URL for a file of data to be used for training.
     */
    dataUrl?: string | null;
    /**
     * Optional - URL for a saved model to load.
     */
    modelUrl?: string | null;
    /**
     * Optional - Custom layer configurations for the underlying TensorFlow layers model.
     */
    layers?: LayerJson[];
    /**
     * Task for the model to perform. Should be one of:
     * 'classification', 'regression', 'imageClassification'
     */
    task: string;
    /**
     * Optional - If true, show graphs while training.
     */
    debug: boolean;
    /**
     * Optional - Learning rate for training the model.
     * If not provided, the default learning rate will depend on the task.
     */
    learningRate?: number;
    /**
     * Optional - Will determine the number of units on the initial layer for 'regression' and 'classification' tasks.
     * Has no impact if the task is 'imageClassification'. Default: 16.
     * TODO: better explanation of what this is
     */
    hiddenUnits?: number;
    /**
     * Optional - If true, create the model layers without training data.
     */
    noTraining?: boolean;
    // could include options for compile.  right now these settings are created from the task.
    /*loss?: string|string[]|{[outputName: string]: string}|LossOrMetricFn|
        LossOrMetricFn[]|{[outputName: string]: LossOrMetricFn};
    metrics?: string|LossOrMetricFn|Array<string|LossOrMetricFn>|
        {[outputName: string]: string | LossOrMetricFn};
       */
}

class DiyNeuralNetwork {
    options: NeuralNetworkOptions;
    neuralNetwork: NeuralNetwork;
    neuralNetworkData: NeuralNetworkData;
    neuralNetworkVis: NeuralNetworkVis;
    data: {
        training: NormalizedData;
    }
    isReady: boolean = false;
    ready: Promise<DiyNeuralNetwork>;
    private task?: NNTask;

    constructor(options: Partial<NeuralNetworkOptions> = {}, callback?: Callback<DiyNeuralNetwork>) {
        // TODO: is is still required if layers are provided?
        if ( ! options.task ) {
            throw new Error(`Missing required option 'task'. Task must be one of ${Object.keys(TASKS).join(', ')}`);
        }

        this.task = findTask(options.task);
        const taskDefaults = this.task?.getDefaultOptions ? this.task.getDefaultOptions() : {};

        this.options = {
            ...DEFAULTS,
            ...taskDefaults,
            ...options,
        };

        this.neuralNetwork = new NeuralNetwork();
        this.neuralNetworkData = new NeuralNetworkData();
        this.neuralNetworkVis = new NeuralNetworkVis();

        this.data = {
            training: [],
        };

        // Initialize
        this.ready = callCallback(this.init(), callback);
    }

    /**
     * ////////////////////////////////////////////////////////////
     * Initialization
     * ////////////////////////////////////////////////////////////
     */

    /**
     * init
     */
    async init(): Promise<this> {
        // check if the a static model should be built based on the inputs and output properties
        if (this.options.noTraining) {
            this.createLayersNoTraining();
        }

        if (this.options.dataUrl) {
            await this.loadDataFromUrl(this.options.dataUrl);
        } else if (this.options.modelUrl) {
            // will take a URL to model.json, an object, or files array
            await this.load(this.options.modelUrl);
        }

        this.isReady = true;
        return this;
    }

    /**
     * createLayersNoTraining
     */
    createLayersNoTraining() {
        // Create sample data based on options
        const {inputs, outputs, task} = this.options;
        if (task === 'classification') {
            for (let i = 0; i < outputs.length; i += 1) {
                const inputSample = new Array(inputs).fill(0);
                this.addData(inputSample, [outputs[i]]);
            }
        } else {
            const inputSample = new Array(inputs).fill(0);
            const outputSample = new Array(outputs).fill(0);
            this.addData(inputSample, outputSample);
        }

        this.neuralNetworkData.createMetadata(this.neuralNetworkData.data.raw);
        if ( ! task ) {
            throw new Error("Property 'task' is missing in options. Cannot create layers without a defined task.");
        }
        this.addDefaultLayers(task, this.neuralNetworkData.meta);
    }

    /**
     * copy
     */
    copy(): DiyNeuralNetwork {
        const nnCopy = new DiyNeuralNetwork(this.options);
        tf.tidy(() => {
            const weights = this.neuralNetwork.model.getWeights();
            const weightCopies = weights.map(weight => weight.clone());
            nnCopy.neuralNetwork.model.setWeights(weightCopies);
        });
        return nnCopy;
    }

    /**
     * ////////////////////////////////////////////////////////////
     * Adding Data
     * ////////////////////////////////////////////////////////////
     */

    /**
     * addData
     * @param {Array | Object} xInputs
     * @param {Array | Object} yInputs
     * @param {*} options
     */
    addData(xInputs, yInputs, options = null) {
        const {inputs, outputs} = this.options;

        // get the input and output labels
        // or infer them from the data
        let inputLabels;
        let outputLabels;

        if (options !== null) {
            // eslint-disable-next-line prefer-destructuring
            inputLabels = options.inputLabels;
            // eslint-disable-next-line prefer-destructuring
            outputLabels = options.outputLabels;
        } else if (inputs.length > 0 && outputs.length > 0) {
            // if the inputs and outputs labels have been defined
            // in the constructor
            if (inputs.every(item => typeof item === 'string')) {
                inputLabels = inputs;
            }
            if (outputs.every(item => typeof item === 'string')) {
                outputLabels = outputs;
            }
        } else if (typeof xInputs === 'object' && typeof yInputs === 'object') {
            inputLabels = Object.keys(xInputs);
            outputLabels = Object.keys(yInputs);
        } else {
            inputLabels = nnUtils.createLabelsFromArrayValues(xInputs, 'input');
            outputLabels = nnUtils.createLabelsFromArrayValues(yInputs, 'output');
        }

        // Make sure that the inputLabels and outputLabels are arrays
        if (!(inputLabels instanceof Array)) {
            throw new Error('inputLabels must be an array');
        }
        if (!(outputLabels instanceof Array)) {
            throw new Error('outputLabels must be an array');
        }

        const formattedInputs = this.searchAndFormat(xInputs);
        const xs = nnUtils.formatDataAsObject(formattedInputs, inputLabels);

        const ys = nnUtils.formatDataAsObject(yInputs, outputLabels);

        this.neuralNetworkData.addData(xs, ys);
    }

    /**
     * loadData
     */
    private loadDataFromUrl(url: string, callback?: Callback<void>): Promise<void> {
        return callCallback(this.loadDataInternal(url), callback);
    }

    /**
     * loadDataInternal
     */
    private async loadDataInternal(url: string): Promise<void> {
        const {inputs, outputs} = this.options;

        const data = await this.neuralNetworkData.loadDataFromUrl(url, inputs, outputs);

        // once the data are loaded, create the metadata
        // and prep the data for training
        // if the inputs are defined as an array of [img_width, img_height, channels]
        this.createMetaData(data);

        this.prepareForTraining(data);
    }

    /**
     * ////////////////////////////////////////////////////////////
     * Metadata prep
     * ////////////////////////////////////////////////////////////
     */

    createMetaData(dataRaw: RawData) {
        const {inputs} = this.options;

        let inputShape;
        if (Array.isArray(inputs) && inputs.length > 0) {
            inputShape =
                inputs.every(item => typeof item === 'number') && inputs.length > 0 ? inputs as number[] : undefined;
        }

        this.neuralNetworkData.createMetadata(dataRaw, inputShape);
    }

    /**
     * ////////////////////////////////////////////////////////////
     * Data prep and handling
     * ////////////////////////////////////////////////////////////
     */

    /**
     * Prepare data for training by applying oneHot to raw
     * @param {*} dataRaw
     */
    private prepareForTraining(dataRaw: RawData = this.neuralNetworkData.data.raw) {
        const unnormalizedTrainingData = this.neuralNetworkData.applyOneHotEncodingsToDataRaw(dataRaw);
        this.data.training = unnormalizedTrainingData;
        this.neuralNetworkData.isWarmedUp = true;

        return unnormalizedTrainingData;
    }

    /**
     * normalizeData
     * @param {*} dataRaw
     * @param {*} meta
     */
    public normalizeData(dataRaw = this.neuralNetworkData.data.raw) {
        if (!this.neuralNetworkData.isMetadataReady) {
            // if the inputs are defined as an array of [img_width, img_height, channels]
            this.createMetaData(dataRaw);
        }

        if (!this.neuralNetworkData.isWarmedUp) {
            this.prepareForTraining(dataRaw);
        }

        const trainingData = this.neuralNetworkData.normalizeDataRaw(dataRaw);

        // set this equal to the training data
        this.data.training = trainingData;

        // set isNormalized to true
        this.neuralNetworkData.meta.isNormalized = true;

        return trainingData;
    }

    /**
     * search though the xInputs and format for adding to data.raws
     * @param {*} input
     */
    private searchAndFormat(input) {
        let formattedInputs;
        if (Array.isArray(input)) {
            formattedInputs = input.map(item => this.formatInputItem(item));
        } else if (typeof input === 'object') {
            const newXInputs = Object.assign({}, input);
            Object.keys(input).forEach(k => {
                const val = input[k];
                newXInputs[k] = this.formatInputItem(val);
            });
            formattedInputs = newXInputs;
        }
        return formattedInputs;
    }

    /**
     * Returns either the original input or a pixelArray[]
     * @param {*} input
     */
    // eslint-disable-next-line class-methods-use-this
    private formatInputItem<T extends any>(input: T): T | Uint8ClampedArray {
        const {image} = new ArgSeparator(input);

        if (image) {
            // TODO: make this not async!!
            return toPixels(image);
        }

        return input;
    }

    /**
     * convertTrainingDataToTensors
     * @param {*} trainingData
     * @param {*} meta
     */
    private convertTrainingDataToTensors(trainingData = this.data.training, meta = this.neuralNetworkData.meta) {
        return this.neuralNetworkData.convertRawToTensors(trainingData, meta);
    }

    /**
     * format the inputs for prediction
     * this means applying onehot or normalization
     * so that the user can use original data units rather
     * than having to normalize
     * @param {*} input
     * @param {*} meta
     * @param {*} inputHeaders
     */
    private formatInputsForPrediction(input: Input, meta: Pick<Metadata, 'inputs'>, inputHeaders: string[]) {

        // TODO: check to see if it is a nested array
        // to run predict or classify on a batch of data

        if (Array.isArray(input)) {
            return inputHeaders.flatMap((prop, idx) =>
                this.isOneHotEncodedOrNormalized(input[idx], prop, meta.inputs)
            );
        } else {
            // TODO: make sure that the input order is preserved!
            return inputHeaders.flatMap(prop =>
                this.isOneHotEncodedOrNormalized(input[prop], prop, meta.inputs)
            );
        }
        // inputData = tf.tensor([inputData.flat()])
    }

    /**
     * formatInputsForPredictionAll
     * @param {*} input
     * @param {*} meta
     * @param {*} inputHeaders
     */
    private formatInputsForPredictionAll(input: Input[] | Input, meta: Pick<Metadata, 'inputs'>, inputHeaders: string[]): tf.Tensor {
        if (input instanceof Array) {
            if (input.every(item => Array.isArray(item))) {
                const output = input.map(item => {
                    return this.formatInputsForPrediction(item, meta, inputHeaders);
                });

                return tf.tensor(output, [input.length, inputHeaders.length]);
            }
            const output = this.formatInputsForPrediction(input, meta, inputHeaders);
            return tf.tensor([output]);
        }

        const output = this.formatInputsForPrediction(input, meta, inputHeaders);
        return tf.tensor([output]);
    }

    /**
     * check if the input needs to be onehot encoded or normalized
     */
    // eslint-disable-next-line class-methods-use-this
    private isOneHotEncodedOrNormalized(input: string | number, key: string, meta: KeyedInputMeta): number | number[] {
        if (typeof input !== 'number') {
            return meta[key].legend[input];
        } else if (this.neuralNetworkData.meta.isNormalized) {
            const {min, max} = meta[key];
            return nnUtils.normalizeValue(input, min, max);
        }
        return input;
    }

    /**
     * ////////////////////////////////////////////////////////////
     * Model prep
     * ////////////////////////////////////////////////////////////
     */

    /**
     * train
     * @param {*} optionsOrCallback
     * @param {*} optionsOrWhileTraining
     * @param {*} callback
     */
    // TODO: actual function types
    public train(optionsOrCallback?: Partial<TrainingOptions> | Function, optionsOrWhileTraining?: Function, callback?: Function) {
        // TODO: can't just use arg separator because there are two functions
        let options;
        let whileTrainingCb;
        let finishedTrainingCb;
        if (
            typeof optionsOrCallback === 'object' &&
            typeof optionsOrWhileTraining === 'function' &&
            typeof callback === 'function'
        ) {
            options = optionsOrCallback;
            whileTrainingCb = optionsOrWhileTraining;
            finishedTrainingCb = callback;
        } else if (
            typeof optionsOrCallback === 'object' &&
            typeof optionsOrWhileTraining === 'function'
        ) {
            options = optionsOrCallback;
            whileTrainingCb = null;
            finishedTrainingCb = optionsOrWhileTraining;
        } else if (
            typeof optionsOrCallback === 'function' &&
            typeof optionsOrWhileTraining === 'function'
        ) {
            options = {};
            whileTrainingCb = optionsOrCallback;
            finishedTrainingCb = optionsOrWhileTraining;
        } else {
            options = {};
            whileTrainingCb = null;
            finishedTrainingCb = optionsOrCallback;
        }

        this.trainInternal(options, whileTrainingCb, finishedTrainingCb);
    }

    /**
     * train
     */
    trainInternal(_options: Partial<TrainingOptions>, whileTrainingCb, finishedTrainingCb) {
        const options = {
            epochs: 10,
            batchSize: 32,
            validationSplit: 0.1,
            whileTraining: null,
            ..._options,
        };

        // if debug mode is true, then use tf vis
        if (this.options.debug) {
            options.whileTraining = [
                this.neuralNetworkVis.trainingVis(),
                {
                    onEpochEnd: whileTrainingCb,
                },
            ];
        } else {
            // if not use the default training
            // options.whileTraining = whileTrainingCb === null ? [{
            //     onEpochEnd: (epoch, loss) => {
            //       console.log(epoch, loss.loss)
            //     }
            //   }] :
            //   [{
            //     onEpochEnd: whileTrainingCb
            //   }];
            options.whileTraining = [
                {
                    onEpochEnd: whileTrainingCb,
                },
            ];
        }

        // if metadata needs to be generated about the data
        if (!this.neuralNetworkData.isMetadataReady) {
            // if the inputs are defined as an array of [img_width, img_height, channels]
            this.createMetaData(this.neuralNetworkData.data.raw);
        }

        // if the data still need to be summarized, onehotencoded, etc
        if (!this.neuralNetworkData.isWarmedUp) {
            this.prepareForTraining(this.neuralNetworkData.data.raw);
        }

        // if inputs and outputs are not specified
        // in the options, then create the tensors
        // from the this.neuralNetworkData.data.raws
        if (!options.inputs && !options.outputs) {
            const {inputs, outputs} = this.convertTrainingDataToTensors();
            options.inputs = inputs;
            options.outputs = outputs;
        }

        // check to see if layers are passed into the constructor
        // then use those to create your architecture
        if (!this.neuralNetwork.isLayered) {
            this.options.layers = this.createNetworkLayers(
                this.options.layers,
                this.neuralNetworkData.meta,
            );
        }

        // if the model does not have any layers defined yet
        // then use the default structure
        if (!this.neuralNetwork.isLayered) {
            this.options.layers = this.addDefaultLayers();
        }

        if (!this.neuralNetwork.isCompiled) {
            // compile the model with defaults
            this.compile();
        }

        // train once the model is compiled
        this.neuralNetwork.train(options, finishedTrainingCb);
    }

    /**
     * addLayer
     * @param {*} options
     */
    private addLayer(options: tf.layers.Layer) {
        this.neuralNetwork.addLayer(options);
    }

    /**
     * Modifies layers by adding inputShape to the first and units (output units) to the last.
     * Adds the layer to the Neural Network.
     */
    private createNetworkLayers(layers: LayerJson[], meta = this.neuralNetworkData.meta): LayerJson[] {
        const {inputUnits, outputUnits} = meta;
        const layersLength = layers.length;

        if (layers.length < 2) {
            throw new Error("Must have at least two layers");
        }

        layers.forEach((layer, i) => {
            // set the inputShape
            const ifFirst = i === 0 ? {inputShape: inputUnits} : {};
            // set the output units
            const ifLast = i ===  layersLength - 1 ? {units: outputUnits} : {};
            const layerObject = tf.layers[layer.type]({...ifFirst, ...ifLast, ...layer} as any);
            this.addLayer(layerObject);
        });

        // TODO: does it need to return the modified layers?
        return layers;
    }

    /**
     * addDefaultLayers
     */
    private addDefaultLayers() {
        const layers = this.task?.createLayers() // TODO: args
        return this.createNetworkLayers(layers);
    }

    /**
     * compile the model
     */
    private compile(modelOptions?: tf.ModelCompileArgs) {
        const {learningRate} = this.options;

        // get options based on the task
        const {optimizer = tf.train.sgd, ...rest} = this.task?.getCompileOptions() || {};

        // bind the optimizer to the model and set the learning rate
        const optimizerFn = this.neuralNetwork.setOptimizerFunction(learningRate, optimizer);

        const options = {
            ...rest,
            optimizer: optimizerFn,
            // override with any provided
            ...modelOptions
        }
        this.neuralNetwork.compile(options);

        // if debug mode is true, then show the model summary
        if (this.options.debug) {
            this.neuralNetworkVis.modelSummary(
                {
                    name: 'Model Summary',
                },
                this.neuralNetwork.model,
            );
        }
    }

    /**
     * ////////////////////////////////////////////////////////////
     * Prediction / classification
     * ////////////////////////////////////////////////////////////
     */

    /**
     * synchronous predict
     */
    private predictSync(input: Input): Prediction[] {
        return this.predictSyncInternal(input);
    }

    /**
     * predict
     */
    public predict(input: Input, callback: Callback<Prediction[]>): Promise<Prediction[]> {
        return callCallback(this.predictInternal(input), callback);
    }

    /**
     * predictMultiple
     */
    public predictMultiple(input: Input[], callback: Callback<Prediction[][]>): Promise<Prediction[][]> {
        return callCallback(this.predictInternal(input), callback);
    }

    /**
     * synchronous classify
     * @param {*} input
     */
    private classifySync(input: Input): LabelAndConfidence {
        return this.classifySyncInternal(input);
    }

    /**
     * classify
     * @param {*} input
     * @param {*} callback
     */
    public classify(input: Input, callback: Callback<LabelAndConfidence>): Promise<LabelAndConfidence> {
        return callCallback(this.classifyInternal(input), callback);
    }

    /**
     * classifyMultiple
     * @param {*} input
     * @param {*} callback
     */
    public classifyMultiple(input: Input[], callback: Callback<LabelAndConfidence[]>): Promise<LabelAndConfidence[]> {
        return callCallback(this.classifyInternal(input), callback);
    }

    /**
     * synchronous predict internal
     */
    private predictSyncInternal(input: Input) {
        const {meta} = this.neuralNetworkData;
        const headers = Object.keys(meta.inputs);

        const inputData = this.formatInputsForPredictionAll(input, meta, headers);

        const unformattedResults = this.neuralNetwork.predictSync(inputData);
        inputData.dispose();

        if (meta !== null) {
            const labels = Object.keys(meta.outputs);

            const formattedResults = unformattedResults.map(unformattedResult => {
                return labels.map((item, idx) => {
                    // check to see if the data were normalized
                    // if not, then send back the values, otherwise
                    // unnormalize then return
                    let val;
                    let unNormalized;
                    if (meta.isNormalized) {
                        const {min, max} = meta.outputs[item];
                        val = nnUtils.unnormalizeValue(unformattedResult[idx], min, max);
                        unNormalized = unformattedResult[idx];
                    } else {
                        val = unformattedResult[idx];
                    }

                    const d = {
                        [labels[idx]]: val,
                        label: item,
                        value: val,
                    };

                    // if unNormalized is not undefined, then
                    // add that to the output
                    if (unNormalized) {
                        d.unNormalizedValue = unNormalized;
                    }

                    return d;
                });
            });

            // return single array if the length is less than 2,
            // otherwise return array of arrays
            if (formattedResults.length < 2) {
                return formattedResults[0];
            }
            return formattedResults;
        }

        // if no meta exists, then return unformatted results;
        return unformattedResults;
    }

    /**
     * predict
     */
    private async predictInternal(input: Input): Promise<Prediction[]>;
    private async predictInternal(input: Input[]): Promise<Prediction[][]>;
    private async predictInternal(input: Input | Input[]): Promise<Prediction[] | Prediction[][]> {
        const {meta} = this.neuralNetworkData;
        const headers = Object.keys(meta.inputs);

        const inputData = this.formatInputsForPredictionAll(input, meta, headers);

        const unformattedResults = await this.neuralNetwork.predict(inputData);
        inputData.dispose();

        if (meta !== null) {
            const labels = Object.keys(meta.outputs);

            const formattedResults = unformattedResults.map(unformattedResult => {
                return labels.map((item, idx) => {
                    // check to see if the data were normalized
                    // if not, then send back the values, otherwise
                    // unnormalize then return
                    let val;
                    let unNormalized;
                    if (meta.isNormalized) {
                        const {min, max} = meta.outputs[item];
                        val = nnUtils.unnormalizeValue(unformattedResult[idx], min, max);
                        unNormalized = unformattedResult[idx];
                    } else {
                        val = unformattedResult[idx];
                    }

                    const d = {
                        [labels[idx]]: val,
                        label: item,
                        value: val,
                    };

                    // if unNormalized is not undefined, then
                    // add that to the output
                    if (unNormalized) {
                        d.unNormalizedValue = unNormalized;
                    }

                    return d;
                });
            });

            // return single array if the length is less than 2,
            // otherwise return array of arrays
            if (formattedResults.length < 2) {
                return formattedResults[0];
            }
            return formattedResults;
        }

        // if no meta exists, then return unformatted results;
        return unformattedResults;
    }

    /**
     * synchronous classify internal
     */
    private classifySyncInternal(input: Input) {
        const {meta} = this.neuralNetworkData;
        const headers = Object.keys(meta.inputs);

        let inputData;

        if (this.options.task === 'imageClassification') {
            // get the inputData for classification
            // if it is a image type format it and
            // flatten it
            inputData = this.searchAndFormat(input);
            if (Array.isArray(inputData)) {
                inputData = inputData.flat();
            } else {
                inputData = inputData[headers[0]];
            }

            if (meta.isNormalized) {
                // TODO: check to make sure this property is not static!!!!
                const {min, max} = meta.inputs[headers[0]];
                inputData = this.neuralNetworkData.normalizeArray(Array.from(inputData), {min, max});
            } else {
                inputData = Array.from(inputData);
            }

            inputData = tf.tensor([inputData], [1, ...meta.inputUnits]);
        } else {
            inputData = this.formatInputsForPredictionAll(input, meta, headers);
        }

        const unformattedResults = this.neuralNetwork.classifySync(inputData);
        inputData.dispose();

        if (meta !== null) {
            const label = Object.keys(meta.outputs)[0];
            const vals = Object.entries(meta.outputs[label].legend);

            const formattedResults = unformattedResults.map(unformattedResult => {
                return vals
                    .map((item, idx) => {
                        return {
                            [item[0]]: unformattedResult[idx],
                            label: item[0],
                            confidence: unformattedResult[idx],
                        };
                    })
                    .sort((a, b) => b.confidence - a.confidence);
            });

            // return single array if the length is less than 2,
            // otherwise return array of arrays
            if (formattedResults.length < 2) {
                return formattedResults[0];
            }
            return formattedResults;
        }

        return unformattedResults;
    }

    /**
     * classify
     */
    private async classifyInternal(input: Input | Input[]): Promise<LabelAndConfidence | LabelAndConfidence[]> {
        const {meta} = this.neuralNetworkData;
        const headers = Object.keys(meta.inputs);

        let inputData;

        if (this.options.task === 'imageClassification') {
            // get the inputData for classification
            // if it is a image type format it and
            // flatten it
            inputData = this.searchAndFormat(input);
            if (Array.isArray(inputData)) {
                inputData = inputData.flat();
            } else {
                inputData = inputData[headers[0]];
            }

            if (meta.isNormalized) {
                // TODO: check to make sure this property is not static!!!!
                const {min, max} = meta.inputs[headers[0]];
                inputData = this.neuralNetworkData.normalizeArray(Array.from(inputData), {min, max});
            } else {
                inputData = Array.from(inputData);
            }

            inputData = tf.tensor([inputData], [1, ...meta.inputUnits]);
        } else {
            inputData = this.formatInputsForPredictionAll(input, meta, headers);
        }

        const unformattedResults = await this.neuralNetwork.classify(inputData);
        inputData.dispose();

        if (meta !== null) {
            const label = Object.keys(meta.outputs)[0];
            const vals = Object.entries(meta.outputs[label].legend);

            const formattedResults = unformattedResults.map(unformattedResult => {
                return vals
                    .map((item, idx) => {
                        return {
                            [item[0]]: unformattedResult[idx],
                            label: item[0],
                            confidence: unformattedResult[idx],
                        };
                    })
                    .sort((a, b) => b.confidence - a.confidence);
            });

            // return single array if the length is less than 2,
            // otherwise return array of arrays
            if (formattedResults.length < 2) {
                return formattedResults[0];
            }
            return formattedResults;
        }

        return unformattedResults;
    }

    /**
     * ////////////////////////////////////////////////////////////
     * Save / Load Data
     * ////////////////////////////////////////////////////////////
     */

    /**
     * save data
     */
    public async saveData(name?: string, callback?: Callback<void>): Promise<void> {
        return callCallback(this.neuralNetworkData.saveData(name), callback);
    }

    /**
     * load data
     * @param {*} filesOrPath
     * @param {*} callback
     */
    public async loadData(filesOrPath: string | FileList, callback?: Callback<void>): Promise<void> {
        return callCallback(this.neuralNetworkData.loadData(filesOrPath), callback);
    }

    /**
     * ////////////////////////////////////////////////////////////
     * Save / Load Model
     * ////////////////////////////////////////////////////////////
     */

    /**
     * saves the model, weights, and metadata
     * @param {*} nameOrCb
     * @param {*} cb
     */
    public async save(nameOrCb?: string | Callback<this>, cb?: Callback<this>): Promise<this> {
        const {string: modelName, callback} = new ArgSeparator(nameOrCb, cb);

        return callCallback(
            Promise.all([
                this.neuralNetwork.save(modelName),
                this.neuralNetworkData.saveMeta(modelName)
            ]).then(() => this),
            callback);
    }

    /**
     * load a model and metadata
     * @param {*} filesOrPath
     * @param {*} callback
     */
    public async load(filesOrPath: string | FileList | { model: string, weights: string; metadata: string }, callback?: Callback<this>): Promise<this> {
        return callCallback(
            Promise.all([
                this.neuralNetwork.load(filesOrPath),
                this.neuralNetworkData.loadMeta(filesOrPath)
            ]).then(() => this),
            callback);
    }

    /**
     * TODO: not documented
     * dispose and release memory for a model
     */
    public dispose(): void {
        this.neuralNetwork.dispose();
    }

    /**
     * ////////////////////////////////////////////////////////////
     * New methods for Neuro Evolution
     * ////////////////////////////////////////////////////////////
     */

    /**
     * TODO: not documented
     * mutate the weights of a model
     * @param {*} rate
     * @param {*} mutateFunction
     */
    public mutate(rate: number, mutateFunction: (value: number) => number): void {
        this.neuralNetwork.mutate(rate, mutateFunction);
    }

    /**
     * TODO: not documented
     * create a new neural network with crossover
     * @param {*} other
     */
    public crossover(other: DiyNeuralNetwork): DiyNeuralNetwork {
        const nnCopy = this.copy();
        nnCopy.neuralNetwork.crossover(other.neuralNetwork);
        return nnCopy;
    }
}

const neuralNetwork = (inputsOrOptions, outputsOrCallback, callback) => {
    let options;
    let cb;

    if (inputsOrOptions instanceof Object) {
        options = inputsOrOptions;
        cb = outputsOrCallback;
    } else {
        options = {
            inputs: inputsOrOptions,
            outputs: outputsOrCallback,
        };
        cb = callback;
    }

    return new DiyNeuralNetwork(options, cb);
};

export default neuralNetwork;
