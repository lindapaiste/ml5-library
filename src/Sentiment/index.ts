import * as tf from '@tensorflow/tfjs';
import {LayersModel} from '@tensorflow/tfjs';
import callCallback, {Callback} from '../utils/callcallback';
import modelLoader from '../utils/modelLoader';
import {loadFile} from "../utils/io";

/**
 * Initializes the Sentiment demo.
 */

const OOV_CHAR = 2;
const PAD_CHAR = 0;

/**
 * Helper function for preparing data
 *
 * @param sequences
 * @param maxLen
 * @param padding
 * @param truncating
 * @param value
 */
function padSequences(sequences: number[][], maxLen: number, padding: string = 'pre', truncating: string = 'pre', value: number = PAD_CHAR): number[][] {
    return sequences.map((seq) => {
        // Perform truncation.
        if (seq.length > maxLen) {
            if (truncating === 'pre') {
                seq.splice(0, seq.length - maxLen);
            } else {
                seq.splice(maxLen, seq.length - maxLen);
            }
        }
        // Perform padding.
        if (seq.length < maxLen) {
            const pad = [];
            for (let i = 0; i < maxLen - seq.length; i += 1) {
                pad.push(value);
            }
            if (padding === 'pre') {
                // eslint-disable-next-line no-param-reassign
                seq = pad.concat(seq);
            } else {
                // eslint-disable-next-line no-param-reassign
                seq = seq.concat(pad);
            }
        }
        return seq;
    });
}


/**
 * Expected format for the metadata.json file
 */
interface MetaData {
    // used in model
    index_from: number;
    max_len: number;
    word_index: Record<string, number>;
    vocabulary_size: number;
    // present in movieReviews metadata.json but not used
    epochs?: number;
    embedding_size?: number;
    model_type?: string;
    batch_size?: number;
}

/**
 * Fully loaded model.
 */
class SentimentModel {
    constructor(public readonly model: LayersModel, public readonly config: MetaData) {
    }

    /**
     * Scores the sentiment of given text with a value between 0 ("negative") and 1 ("positive").
     * @param {string} text - string of text to predict.
     * @returns {{score: number}}
     */
    predict(text: string): { score: number } {
        const {word_index, index_from, max_len, vocabulary_size} = this.config;

        // Convert to lower case and remove all punctuations.
        const inputWords =
            text.trim().toLowerCase().replace(/[.,?!]/g, '').split(' ');

        // Convert the words to a sequence of word indices.
        const sequence = inputWords.map((word) => {
            let wordIndex = word_index[word] + index_from;
            if (wordIndex > vocabulary_size) {
                wordIndex = OOV_CHAR;
            }
            return wordIndex;
        });

        // Perform truncation and padding.
        const paddedSequence = padSequences([sequence], max_len);
        const input = tf.tensor2d(paddedSequence, [1, max_len]);
        const predictOut = this.model.predict(input) as tf.Tensor;
        const score = predictOut.dataSync()[0];
        predictOut.dispose();
        input.dispose();

        return {
            score
        };
    }

}

/**
 * Asynchronously load the Sentiment model.
 *
 * @param {string} modelName
 * @return {Promise<SentimentModel>}
 */
const loadSentimentModel = async (modelName: string): Promise<SentimentModel> => {
    let directory = modelName;
    if (modelName.toLowerCase() === 'moviereviews') {
        directory = 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1';
    }
    // TODO: review/create logic for calling with custom model file path
    // Current logic supports passing the name of a directory with a model.json and a metadata.json
    // Or a path to a model.json which has a metadata.json in the same directory.
    const loader = modelLoader(directory);
    const modelUrl = loader.modelJsonPath();
    //throw new Error(`${modelName} is not a valid model name or path.`);

    const model = await tf.loadLayersModel(modelUrl);
    const metadata = await loadFile<MetaData>(loader.fileInDirectory("metadata.json"));

    return new SentimentModel(model, metadata);
}

/**
 * Create Sentiment model. Currently the supported model name is 'moviereviews'. ml5 may support different models in the future.
 * @param {string} modelName - A string to the path of the JSON model.
 * @param {function} [callback] - Optional. A callback function that is called once the model has loaded.
 * If no callback is provided, it will return a promise that will be resolved once the model has loaded.
 */
const sentiment = (modelName: string, callback: Callback<SentimentModel>): Promise<SentimentModel> =>
    callCallback(loadSentimentModel(modelName), callback);

export default sentiment;