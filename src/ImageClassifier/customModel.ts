import * as tf from "@tensorflow/tfjs";
import {imgToTensor, TfImageSource} from "../utils/imageUtilities";
import {ClassifierModel, ClassifierModelConfig} from "./darknet";
import {Classification, getTopKClassesFromArray} from "../utils/gettopkclasses";
import {LayersModel, Tensor} from "@tensorflow/tfjs";
import {loadFile} from "../utils/io";

export class CustomModel implements ClassifierModel {

    constructor(public readonly model: LayersModel, private config: ClassifierModelConfig) {
    }

    async classify(img: TfImageSource, topk: number): Promise<Classification[]> {
        //await tf.nextFrame();
            const predictedClasses = tf.tidy(() => {
                const processedImg = imgToTensor(img, [this.config.imgSize, this.config.imgSize]);
                const predictions = this.model.predict(processedImg) as Tensor;
                return Array.from(predictions.as1D().dataSync());
            });
            return getTopKClassesFromArray(predictedClasses, topk, this.config.classes);
        }
}

// TODO: clean up and combine with other model loaders
const loadClasses = async (path: string): Promise<string[] | Record<number, string>> => {
    const data = await loadFile(path);

    if (data.ml5Specs && data.ml5Specs.mapStringToIndex) {
        return data.ml5Specs.mapStringToIndex;
    }

    const split = path.split("/");
    const prefix = split.slice(0, split.length - 1).join("/");
    const metadataUrl = `${prefix}/metadata.json`;

    let metadata;
    try {
        metadata = await loadFile(metadataUrl);
    } catch (error) {
        throw new Error(`Tried to fetch metadata.json from URL ${metadataUrl} but it seems to be missing. Received Error: ${error?.message}`);
    }
    if ( metadata && metadata.labels ) {
        return metadata.labels;
    }
    throw new Error(`metadata.json file is missing the required property 'labels'`);
}

// TODO: where does image size come from? Is it an argument or is it in the metadata.json response?
const IMAGE_SIZE = 224;

export const load = async (path: string): Promise<ClassifierModel> => {
    const classes = await loadClasses(path);
    const model = await tf.loadLayersModel(path);
    return new CustomModel(model, {
        classes,
        imgSize: IMAGE_SIZE,
        url: path,
    });
}