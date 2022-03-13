import * as toxicity from '@tensorflow-models/toxicity';
import { TextModelConstructor } from "../types";

export interface TextToxicityOptions {
    /**
     * A prediction is considered valid only if its confidence
     * exceeds the threshold. Defaults to 0.85.
     */
    threshold?: number;
    /**
     * An array of strings indicating which types of toxicity
     * to detect. Labels must be one of `toxicity` | `severe_toxicity` |
     * `identity_attack` | `insult` | `threat` | `sexual_explicit` | `obscene`.
     * Defaults to all labels.
     */
    toxicityLabels?: string[];
}

export interface ToxicityClassification {
    label: string;
    results: Array<{
        probabilities: Float32Array;
        match: boolean;
    }>;
}

const createTextToxicity: TextModelConstructor<ToxicityClassification[], TextToxicityOptions> = (initialOptions = {}) => {
    const { threshold, toxicityLabels } = initialOptions;
    let model: toxicity.ToxicityClassifier | null = null;
    return {
        name: 'TextToxicity',
        event: 'classify',
        classify: async (text, options) => {
           // TODO: apply changed options
            if (!model || options) {
                // @ts-ignore -- package incorrectly marks properties as required.
                model = await toxicity.load(threshold, toxicityLabels);
            }
            return model.classify(text);
        },
        defaultOptions: {}
    }
}

export default createTextToxicity;
