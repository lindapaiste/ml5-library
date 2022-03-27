import * as tf from '@tensorflow/tfjs';
import {LayerJson, NeuralNetworkOptions} from "./index";
import {basicParseInputsOutputs, LabelConfig} from "./labels";

/**
 * Separate all task-dependent logic into separate task classes to minimize if/else behavior
 * in the main Neural Network class and make it easier to potentially add more tasks in the future.
 */

// optimizer should be a function that creates an optimizer rather than the optimizer itself
export type CompileOptions = Omit<tf.ModelCompileArgs, 'optimizer'> & {
    // note: learningRate is always the first arg, but some optimizers support other optional args as well
    optimizer: (learningRate: number) => tf.Optimizer;
}

export interface NNTask {
    name: string;

    // can optionally override the standard defaults with custom defaults
    getDefaultOptions?(): Partial<NeuralNetworkOptions>;

    getCompileOptions(): CompileOptions;

    createLayers(inputShape: tf.Shape, hiddenUnits: number, outputUnits: number): LayerJson[];

    parseInputs(provided: number | string[] | number[]): LabelConfig;

    parseOutputs(provided: number | string[] | number[]): LabelConfig;
}

// TODO: should a task take the nn as an argument in the constructor? Or should args be passed to methods?

class ClassificationTask implements NNTask {
    public readonly name: string = 'classification';

    public getCompileOptions(): CompileOptions {
        return {
            loss: 'categoricalCrossentropy',
            optimizer: tf.train.sgd,
            metrics: ['accuracy'],
        };
    }

    public createLayers(inputShape: tf.Shape, hiddenUnits: number, outputUnits: number): LayerJson[] {
        return [
            {
                type: 'dense',
                units: hiddenUnits,
                activation: 'relu',
                inputShape
            },
            {
                type: 'dense',
                activation: 'softmax',
                units: outputUnits,
            },
        ];
    }

    /**
     * Based on current examples, the 'inputs' option for a classification task is either:
     * string[] - The input property names.
     * number[] - The shape of a single input property. For images, it is [width, height, channels].
     * number - The number of input properties.
     */
    parseInputs = basicParseInputsOutputs;


    /**
     * Based on current examples, the 'outputs' option for a classification task is either:
     * [string] - An array with a single string is the property name to use as the label.
     * string[] - An array with multiple strings are the label values. Could also support numeric labels number[].
     * number - A number is the number of options for the label.
     *
     * There is never more than one output property.
     */
    parseOutputs(provided: number | string[] | number[]): LabelConfig {
        if (Array.isArray(provided)) {
            if (provided.length === 1) {
                // A single element must be the property name.
                return {
                    count: 1,
                    names: [provided[0].toString()]
                }
            } else {
                // An array of possible values.
                // TODO: can create a handler from labels, but where to use this?
                return {
                    count: 1,
                    shape: [provided.length],
                }
            }
        } else {
            // The number of options for the label.
            return {
                count: 1,
                shape: [provided],
            }
        }
    }
}

class ImageClassificationTask extends ClassificationTask {
    public readonly name = 'imageClassification';

    getDefaultOptions(): Partial<NeuralNetworkOptions> {
        return {
            learningRate: 0.02,
        }
    }

    // compileOptions are the same

    public createLayers(inputShape: tf.Shape, hiddenUnits: number, outputUnits: number): LayerJson[] {
        return [
            {
                type: 'conv2d',
                filters: 8,
                kernelSize: 5,
                strides: 1,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                inputShape,
            },
            {
                type: 'maxPooling2d',
                poolSize: [2, 2],
                strides: [2, 2],
            },
            {
                type: 'conv2d',
                filters: 16,
                kernelSize: 5,
                strides: 1,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
            },
            {
                type: 'maxPooling2d',
                poolSize: [2, 2],
                strides: [2, 2],
            },
            {
                type: 'flatten',
            },
            {
                type: 'dense',
                kernelInitializer: 'varianceScaling',
                activation: 'softmax',
                units: outputUnits,
            },
        ];
    }
}

class RegressionTask implements NNTask {
    public readonly name = 'regression';

    public getCompileOptions(): CompileOptions {
        return {
            loss: 'meanSquaredError',
            optimizer: tf.train.adam,
            metrics: ['accuracy'],
        };
    }

    public createLayers(inputShape: tf.Shape, hiddenUnits: number, outputUnits: number): LayerJson[] {
        return [
            {
                type: 'dense',
                units: hiddenUnits,
                activation: 'relu',
                inputShape
            },
            {
                type: 'dense',
                activation: 'sigmoid',
                units: outputUnits,
            },
        ];
    }

    parseInputs = basicParseInputsOutputs;

    /**
     * The 'outputs' for a regression task is typically the number 1 (one output property)
     * or an array with one string element (the output property name).
     * TODO: is a number more than 1 supported? What does it mean? See particle example.
     */
    parseOutputs = basicParseInputsOutputs;
}

/**
 * mapping of supported task class constructors and their task names
 */
export const TASKS = {
    regression: RegressionTask,
    classification: ClassificationTask,
    imageClassification: ImageClassificationTask,
}

/**
 * Construct a task instance of the correct class based on the task name.
 */
export default (name: string): NNTask => {
    const match = Object.entries(TASKS).find(([key]) => key.toLowerCase() === name.toLowerCase());
    if (!match) {
        throw new Error(`Unknown task name '${name}'. Task must be one of ${Object.keys(TASKS).join(', ')}`)
    }
    const [_, constructor] = match;
    return new constructor();
}