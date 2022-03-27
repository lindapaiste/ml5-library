import * as tf from "@tensorflow/tfjs";
import {io} from "@tensorflow/tfjs";
import {loadFile, saveBlob} from "../utils/io";
import {Metadata, RawPropertyData} from "./NeuralNetworkData";
import callCallback, {Callback} from "../utils/callcallback";
import {ArgSeparator} from "../utils/argSeparator";
import {createDateFileName, directory, isFileList} from "../utils/modelLoader";
import {XandYRowDataAdder} from "./data";
import {csvToJSON} from "../utils/csv";


/**
 * Neural Network involves up to 4 different files:
 *
 * TensorFlow core:
 * model: 'model/model.json'
 * weights: 'model/model.weights.bin'
 *
 * ml5 added:
 * metadata: 'model/model_meta.json'
 * data: 'data/colorData.csv'
 */

interface ModelLoadPaths {
    model: string;
    weights: string;
    metadata: string;
}

type FileKeys = keyof ModelLoadPaths | "data";

class ModelIO {

    // placeholders for what to do with the actual results:
    onLoadModel(model: tf.LayersModel): void {
    }

    onLoadMeta(meta: Metadata): void {
    }

    /**
     * Saves metadata of the data
     */
    public async saveMeta(modelName: string = 'model'): Promise<void> {
        await saveBlob(JSON.stringify(this.meta), `${modelName}_meta.json`, 'text/plain');
    }

    /**
     * Separates the files from a FileList into known properties based on the file name.
     * @param name
     * @private
     */
    private static assignFileByName(name: string): FileKeys {
        // Note: could add underscores to be stricter.
        if (name.includes('weights') || name.includes('.bin')) {
            return "weights";
        } else if (name.includes('meta')) {
            return "metadata";
        } else if (name.includes('data')) {
            return "data";
        } else if (name.includes('model')) {
            return "model";
        } else {
            throw new Error(`Unknown file name: ${name}. Cannot assign file as any of: model, metadata, weights, data.`);
        }
    }


    /**
     * save the model
     * @param {*} nameOrCb
     * @param {*} cb
     */
    save_fromModel = async (nameOrCb?: string | Callback<any>, cb?: Callback<tf.io.SaveResult>): Promise<tf.io.SaveResult> => {
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
     * Loads model, weights, and metadata.
     * @param {*} filesOrPath
     * @param {*} callback
     */
        // TODO: clean this up
    load_fromModel = async (filesOrPath: string | FileList | ModelLoadPaths, callback?: Callback<tf.Sequential>): Promise<tf.Sequential> => {
        let pathOrIOHandler: string | io.IOHandler;

        if (isFileList(filesOrPath)) {
            const array = Array.from(filesOrPath);
            const modelFile = array.find(file => ModelIO.assignFileByName(file.name) === "model");
            const metaFile = array.find(file => ModelIO.assignFileByName(file.name) === "metadata");
            // TF allows for multiple weights files, so use filter instead of find
            const weightsFiles = array.filter(file => ModelIO.assignFileByName(file.name) === "weights");
            if (!modelFile) {
                throw new Error("No model file found in FileList. Expected a file with name 'model.json'");
            }
            // TODO: Throw an error if no weights are found? Or are they optional? TF considers them optional.
            pathOrIOHandler = tf.io.browserFiles([modelFile, ...weightsFiles]);
            // TODO: Throw an error if no meta? Or is it optional?
            if (metaFile) {
                const metaText = await metaFile.text();
                this.onLoadMeta(JSON.parse(metaText));
            }
        } else if (typeof filesOrPath === "object") {
            // load the modelJson
            const modelJson = await loadFile(filesOrPath.model);
            // TODO: browser File() API won't be available in node env
            const modelJsonFile = new File([JSON.stringify(modelJson)], 'model.json', {type: 'application/json'});

            // load the weights
            const weightsBlob = await loadFile(filesOrPath.weights, 'blob');
            // TODO: browser File() API won't be available in node env
            const weightsBlobFile = new File([weightsBlob], 'model.weights.bin', {
                type: 'application/macbinary', // TODO: is this the right type?
            });

            pathOrIOHandler = tf.io.browserFiles([modelJsonFile, weightsBlobFile]);

            // load the meta
            const meta = await loadFile(filesOrPath.metadata);
            this.onLoadMeta(meta);
        } else {
            pathOrIOHandler = filesOrPath;
            const metaPath = `${directory(filesOrPath)}/model_meta.json`;
            const meta = await loadFile(metaPath);
            this.onLoadMeta(meta);
        }
        // TODO: does it actually create a model of the Sequential type?
        const model = await tf.loadLayersModel(pathOrIOHandler) as tf.Sequential;
        this.onLoadModel(model);
    }
}

/**
 * Handles loading and saving data from JSON files and other sources.
 */
class DataIO {

    constructor(private readonly data: XandYRowDataAdder) {
    }

    public async save(name?: string): Promise<void> {
        const fileName = name || `data_${createDateFileName()}`;

        const output = {
            data: this.data.prepareForSave(),
        };

        await saveBlob(JSON.stringify(output), `${fileName}.json`, 'text/plain');
    }


    /**
     * Public loadData function accepts a URL string or a FileList from an input element.
     */
    public async load(filesOrPath: string | FileList): Promise<void> {
        if (typeof filesOrPath === "string") {
            await this.loadDataFromUrl(filesOrPath);
        } else {
            await this.loadDataFromFileList(filesOrPath);
        }
    }

    /**
     * Loads data from a URL using the appropriate function
     */
    private async loadDataFromUrl(dataUrl: string): Promise<void> {
        if (dataUrl.endsWith('.csv')) {
            return this.loadCSV(dataUrl);
        } else if (dataUrl.endsWith('.json')) {
            return this.loadJSON(dataUrl);
        } else if (dataUrl.includes('blob')) {
            return this.loadBlob(dataUrl);
        } else {
            throw new Error('Not a valid data format. Must be CSV or JSON');
        }
    }

    /**
     * When using a FileList, assume that the file is in JSON or CSV format.
     */
    private async loadDataFromFileList(files: FileList): Promise<void> {
        // TODO: can there be multiple data files?
        const file = files[0];
        if (!file) {
            throw new Error("No files found in FileList");
        }
        const text = await file.text();
        const json = file.name.endsWith('.csv') ? csvToJSON(text) : JSON.parse(text);
        this.addJsonData(json);
    }

    /**
     * Find the array of entries and add them to the data.
     */
    private addJsonData(json: object): void {
        const rows = this.findEntries(json);
        this.data.addManyRows(rows);
    }


    /**
     * Load JSON from a file by its URL string.
     */
    private async loadJSON(url: string): Promise<void> {
        const json = await loadFile(url);
        this.addJsonData(json);
    }


    /**
     * Load data from a CSV through the 'dataUrl' property of the neural network options.
     * Can extract only certain columns if the labels are known.
     * But should infer the labels based on the columns if they are not provided through the options.
     */
    private async loadCSV(dataUrl: string): Promise<void> {
        try {
            let csvConfig;
            // can pass input/output columns to TensorFlow to get rows in the format {xs: features, ys: labels}
            // but this might not be particularly helpful
            /*const inputLabels = this.labels.xs.names;
            const outputLabels = this.labels.ys.names;
            if (inputLabels && outputLabels) {
                const columnConfigs = Object.fromEntries([
                    ...inputLabels.map(name => [name, {isLabel: false, required: true}]),
                    ...outputLabels.map(name => [name, {isLabel: true, required: true}]),
                ]);
                csvConfig = {
                    columnConfigs,
                    configuredColumnsOnly: true,
                }
            }*/
            // if labels are not known, get the whole row as a keyed object
            // assigning to xs/ys will be done in the data handler
            const csvDataset = tf.data.csv(dataUrl, csvConfig);
            const rows = await csvDataset.toArray();
            this.data.addManyRows(rows as any);
        } catch (err) {
            throw new Error(`Error loading CSV: ${err?.message}`);
        }
    }

    /**
     * Use axios to resolve a blob URL.
     */
    private async loadBlob(url: string): Promise<void> {
        const data = await loadFile(url, 'blob');
        // TODO: is it already parsed as JSON?
        // https://stackoverflow.com/questions/59465413/convert-blob-url-to-file-object-with-axios
        try {
            const json = JSON.parse(data);
            this.addJsonData(json);
        } catch (error) {
            // CSV is only a last-ditch attempt to parse data. JSON is expected.
            try {
                const json = csvToJSON(data);
                this.addJsonData(json);
            } catch (csvError) {
                throw new Error(`Cannot parse Blob data as JSON. Error: ${error?.message}`);
            }
        }
    }

    /**
     * Try to extract an array of data points from a JSON object.
     * Expect it to be an object with a property `data` or `entries`.
     * Looks for the array recursively. <-- TODO: is this needed?
     */
    private findEntries(data: Record<string, any> | any[]): Record<string, RawPropertyData>[] | RawPropertyData[][] {
        if (Array.isArray(data)) {
            return data;
        }
        if (data.entries) {
            return this.findEntries(data.entries);
        }
        if (data.data) {
            return this.findEntries(data.entries);
        }

        Object.keys(data).forEach(k => {
            if (typeof data[k] === 'object') {
                // don't throw an error on each property, only throw if none contain data.
                try {
                    return this.findEntries(data[k]);
                } catch (e) {
                }
            }
        });

        throw new Error("Could not find data. JSON is expected to be an object with a property 'data' containing an array of data objects.");
    }

}