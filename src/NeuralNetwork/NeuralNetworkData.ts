import * as tf from '@tensorflow/tfjs';
import axios from 'axios';
import {loadFile, saveBlob} from '../utils/io';
import handleArguments from "../utils/handleArguments";
import { saveBlob } from '../utils/io';
import nnUtils from './NeuralNetworkUtils';
import callCallback, {Callback} from "../utils/callcallback";
import {ArgSeparator} from "../utils/argSeparator";
import {Label} from "./index";

export enum DType {
  ARRAY = 'array',
  STRING = 'string',
  NUMBER = 'number',
}

export type RawPropertyData = number | string | number[];

export type RecursiveRawPropertyData = number | string | Array<RecursiveRawPropertyData>;

export type RawData = {
  xs: Record<string, RawPropertyData>;
  ys: Record<string, RawPropertyData>;
}[];
export type RawDataRow = RawData[number];

export type AddableData = Array<RawDataRow | RawPropertyData[] | Record<string, RawPropertyData>>

export type UnlabeledData = {
  xs: RawPropertyData[];
  ys: RawPropertyData[];
}[];

export type XsOrYs = 'xs' | 'ys';

export type NormalizedData = {
  xs: number | number[] | number[][];
  ys: number | number[] | number[][];
}[];

export type Legend = Record<string, number[]>; // lookup for string OneHot encoding

interface BaseInputMeta {
  min: number;
  max: number;
  dtype: string;
  legend?: Legend;
  uniqueValues?: string[];
}

export type InputMeta = BaseInputMeta & ({
  dtype: DType.ARRAY | DType.NUMBER;
} | {
  dtype: DType.STRING;
  legend: Legend;
  uniqueValues: string[];
})

export type KeyedInputMeta = Record<string, InputMeta>;

export interface Metadata {
  inputUnits: number[]; // despite the name "units" this is actually used as the input shape, so it's an array instead of a number
  outputUnits: number;
  // objects describing input/output data by property name
  inputs: KeyedInputMeta;
  outputs: KeyedInputMeta;
  // whether of not numeric data is normalized
  isNormalized: boolean;
}

class NeuralNetworkData {
  public meta: Metadata;
  public isMetadataReady: boolean;
  public isWarmedUp: boolean;
  public data: {
    raw: RawData;
  }

  constructor() {
    this.meta = {
      inputUnits: [], // was previously null, // Number
      outputUnits: 0, // was previously null, // Number
      // objects describing input/output data by property name
      inputs: {}, // { name1: {dtype}, name2: {dtype}  }
      outputs: {}, // { name1: {dtype} }
      isNormalized: false, // Boolean - keep this in meta for model saving/loading
    };

    this.isMetadataReady = false;
    this.isWarmedUp = false;

    this.data = {
      raw: [], // array of {xs:{}, ys:{}}
    };
  }

  /**
   * ////////////////////////////////////////////////////////
   * Summarize Data
   * ////////////////////////////////////////////////////////
   */

  /**
   * create the metadata from the data
   * this covers:
   *  1. getting the datatype from the data
   *  2. getting the min and max from the data
   *  3. getting the oneHot encoded values
   *  4. getting the inputShape and outputUnits from the data
   * @param {*} dataRaw
   * @param {*} inputShape
   */
  public createMetadata(dataRaw: RawData, inputShape?: number[]) {
    // get the data type for each property
    this.getDTypesFromData(dataRaw);
    // get the stats - min, max
    this.getDataStats(dataRaw);
    // onehot encode
    this.getDataOneHot(dataRaw);
    // calculate the input units from the data
    this.getDataUnits(dataRaw, inputShape);

    this.isMetadataReady = true;
    return { ...this.meta };
  }

  /*
   * ////////////////////////////////////////////////
   * data Summary
   * ////////////////////////////////////////////////
   */

  /**
   * get stats about the data
   * @param {*} dataRaw
   */
  private getDataStats(dataRaw: RawData): Metadata {
    const meta = Object.assign({}, this.meta);

    const inputMeta = this.getInputMetaStats(dataRaw, meta.inputs, 'xs');
    const outputMeta = this.getInputMetaStats(dataRaw, meta.outputs, 'ys');

    meta.inputs = inputMeta;
    meta.outputs = outputMeta;

    this.meta = {
      ...this.meta,
      ...meta,
    };

    return meta;
  }

  /**
   * getRawStats
   * get back the min and max of each label
   * @param {*} dataRaw
   * @param {*} inputOrOutputMeta
   * @param {*} xsOrYs
   */
  // eslint-disable-next-line no-unused-vars, class-methods-use-this
  private getInputMetaStats(dataRaw: RawData, inputOrOutputMeta: KeyedInputMeta, xsOrYs: 'xs' | 'ys') : KeyedInputMeta{
    // TODO: there is no point to cloning a shallow copy
    const inputMeta = Object.assign({}, inputOrOutputMeta);

    Object.keys(inputMeta).forEach(k => {
      if (inputMeta[k].dtype === 'string') {
        inputMeta[k].min = 0;
        inputMeta[k].max = 1;
      } else if (inputMeta[k].dtype === 'number') {
        // note: have to assert type because there is no relationship between the dtype in inputMeta and the dataRaw type
        // could alternatively parse to a number -- might be needed with CSV data
        const dataAsArray = dataRaw.map(item => item[xsOrYs][k]);
        inputMeta[k].min = nnUtils.arrayMin(dataAsArray as number[]);
        inputMeta[k].max = nnUtils.arrayMax(dataAsArray as number[]);
      } else if (inputMeta[k].dtype === 'array') {
        const dataAsArray = dataRaw.map(item => item[xsOrYs][k]).flat();
        inputMeta[k].min = nnUtils.arrayMin(dataAsArray as number[]);
        inputMeta[k].max = nnUtils.arrayMax(dataAsArray as number[]);
      }
    });

    return inputMeta;
  }

  /**
   * get the data units, inputshape and output units
   * @param {*} dataRaw
   */
  private getDataUnits(dataRaw: RawData, arrayShape?: number[]) {
    const meta = Object.assign({}, this.meta);

    // if the data has a shape pass it in
    let inputShape;
    if (arrayShape) {
      inputShape = arrayShape;
    } else {
      inputShape = [this.getInputMetaUnits(dataRaw, meta.inputs)].flat();
    }

    const outputShape = this.getInputMetaUnits(dataRaw, meta.outputs);

    meta.inputUnits = inputShape;
    meta.outputUnits = outputShape;

    this.meta = {
      ...this.meta,
      ...meta,
    };

    return meta;
  }

  /**
   * get input
   * @param {*} _inputsMeta
   * @param {*} dataRaw
   */
  private getInputMetaUnits(dataRaw: RawData, _inputsMeta: KeyedInputMeta): number {
    let units = 0;
    const inputsMeta = Object.assign({}, _inputsMeta);

    Object.entries(inputsMeta).forEach(arr => {
      const { dtype } = arr[1];
      if (dtype === 'number') {
        units += 1;
      } else if (dtype === 'string') {
        const { uniqueValues } = arr[1];

        const uniqueCount = uniqueValues.length;
        units += uniqueCount;
      } else if (dtype === 'array') {
        // TODO: User must input the shape of the
        // image size correctly.
        units = [];
      }
    });

    return units;
  }

  /**
   * getDTypesFromData
   * gets the data types of the data we're using
   * important for handling oneHot
   */
  private getDTypesFromData(dataRaw: RawData) {
    const meta = {
      ...this.meta,
      inputs: {},
      outputs: {},
    };

    const sample = dataRaw[0];
    const xs = Object.keys(sample.xs);
    const ys = Object.keys(sample.ys);

    xs.forEach(prop => {
      meta.inputs[prop] = {
        dtype: nnUtils.getDataType(sample.xs[prop]),
      };
    });

    ys.forEach(prop => {
      meta.outputs[prop] = {
        dtype: nnUtils.getDataType(sample.ys[prop]),
      };
    });

    // TODO: check if all entries have the same dtype.
    // otherwise throw an error

    this.meta = meta;

    return meta;
  }

  /**
   * ////////////////////////////////////////////////////////
   * Add Data
   * ////////////////////////////////////////////////////////
   */

  /**
   * Add Data
   * @param {object} xInputObj, {key: value}, key must be the name of the property value must be a String, Number, or Array
   * @param {*} yInputObj, {key: value}, key must be the name of the property value must be a String, Number, or Array
   */
  public addData(xInputObj: Record<string, number | string | number[]>, yInputObj: Record<string, number | string | number[]>): void {
    this.data.raw.push({
      xs: xInputObj,
      ys: yInputObj,
    });
  }

  /**
   * ////////////////////////////////////////////////////////
   * Tensor handling
   * ////////////////////////////////////////////////////////
   */

  /**
   * convertRawToTensors
   * converts array of {xs, ys} to tensors
   */
  public convertRawToTensors(dataRaw: RawData) {
    const dataLength = dataRaw.length;

    return tf.tidy(() => {
      const inputArr: (string | number)[][] = [];
      const outputArr: (string | number)[][] = [];

      dataRaw.forEach(row => {
        // get xs
        const xs = Object.keys(this.meta.inputs)
          .flatMap(k => {
            return row.xs[k];
          });

        inputArr.push(xs);

        // get ys
        const ys = Object.keys(this.meta.outputs)
          .flatMap(k => {
            return row.ys[k];
          });

        outputArr.push(ys);
      });

      const inputs = tf.tensor(inputArr.flat(), [dataLength, ...this.meta.inputUnits]);
      const outputs = tf.tensor(outputArr.flat(), [dataLength, this.meta.outputUnits]);

      return {
        inputs,
        outputs,
      };
    });
  }

  /**
   * ////////////////////////////////////////////////////////
   * data normalization / unnormalization
   * ////////////////////////////////////////////////////////
   */

  /**
   * normalize the dataRaw input
   * @param {*} dataRaw
   */
  public normalizeDataRaw(dataRaw: RawData): NormalizedData {
    const meta = Object.assign({}, this.meta);

    const normXs = this.normalizeInputData(dataRaw, meta.inputs, 'xs');
    const normYs = this.normalizeInputData(dataRaw, meta.outputs, 'ys');

    return nnUtils.zipArrays(normXs, normYs);
  }

  /**
   * normalizeRaws
   * @param {*} dataRaw
   * @param {*} inputOrOutputMeta
   * @param {*} xsOrYs
   */
  private normalizeInputData(dataRaw: RawData, inputOrOutputMeta: KeyedInputMeta, xsOrYs: 'xs' | 'ys') {
    // the data length
    const dataLength = dataRaw.length;
    // the copy of the inputs.meta[inputOrOutput]
    const inputMeta = Object.assign({}, inputOrOutputMeta);

    // normalized output object
    const normalized = {};
    Object.keys(inputMeta).forEach(k => {
      // get the min and max values
      const options = {
        min: inputMeta[k].min,
        max: inputMeta[k].max,
      };

      const dataAsArray = dataRaw.map(item => item[xsOrYs][k]);
      // depending on the input type, normalize accordingly
      if (inputMeta[k].dtype === 'string') {
        options.legend = inputMeta[k].legend;
        normalized[k] = this.normalizeArray(dataAsArray, options);
      } else if (inputMeta[k].dtype === 'number') {
        normalized[k] = this.normalizeArray(dataAsArray, options);
      } else if (inputMeta[k].dtype === 'array') {
        normalized[k] = dataAsArray.map(item => this.normalizeArray(item, options));
      }
    });

    // create a normalized version of data.raws
    return [...new Array(dataLength)].map((item, idx) => {
      const row = {
        [xsOrYs]: {} as Record<string, any>,
      };

      Object.keys(inputMeta).forEach(k => {
        row[xsOrYs][k] = normalized[k][idx];
      });

      return row;
    });
  }

  /**
   * normalizeArray
   * @param {*} inputArray
   * @param {*} options
   */
  public normalizeArray(inputArray: (string | number)[], options: InputMeta) {
    const { min, max, legend } = options;

    // if the data are onehot encoded, replace the string
    // value with the onehot array
    // if none exists, return the given value
    if (legend) {
      return inputArray.map(v => legend[v] ? legend[v] : v);
    }

    // if the dtype is a number
    if (inputArray.every(v => typeof v === 'number')) {
      return inputArray.map(v => nnUtils.normalizeValue(v, min, max));
    }

    // otherwise return the input array
    // return inputArray;
    throw new Error('error in inputArray of normalizeArray() function');
  }

  /**
   * unNormalizeArray
   * @param {*} inputArray
   * @param {*} options
   */
  private unnormalizeArray(inputArray: number[] | number[][], options: InputMeta) {
    const { min, max } = options;

    // if the data is onehot encoded then remap the
    // values from those oneHot arrays
    if (options.legend) {
      return inputArray.map(v => {
        let res;
        Object.entries(options.legend!).forEach(([key, val]) => {
          const matches = v.map((num, idx) => num === val[idx]).every(truthy => truthy === true);
          if (matches) res = key;
        });
        return res;
      });
    }

    // if the dtype is a number
    if (inputArray.every(v => typeof v === 'number')) {
      return inputArray.map(v => nnUtils.unnormalizeValue(v, min, max));
    }

    // otherwise return the input array
    // return inputArray;
    throw new Error('error in inputArray of normalizeArray() function');
  }

  /*
   * ////////////////////////////////////////////////
   * One hot encoding handling
   * ////////////////////////////////////////////////
   */

  /**
   * applyOneHotEncodingsToDataRaw
   * does not set this.data.raws
   * but rather returns them
   * @param {*} dataRaw
   * @param {*} meta
   */
  public applyOneHotEncodingsToDataRaw(dataRaw: RawData) {
    const meta = Object.assign({}, this.meta);

    return dataRaw.map(row => {
      const xs = {
        ...row.xs,
      };
      const ys = {
        ...row.ys,
      };
      // get xs
      Object.keys(meta.inputs).forEach(k => {
        if (meta.inputs[k].legend) {
          xs[k] = meta.inputs[k].legend[row.xs[k]];
        }
      });

      Object.keys(meta.outputs).forEach(k => {
        if (meta.outputs[k].legend) {
          ys[k] = meta.outputs[k].legend[row.ys[k]];
        }
      });

      return {
        xs,
        ys,
      };
    });
  }

  /**
   * getDataOneHot
   * creates onehot encodings for the input and outputs
   * and adds them to the meta info
   * @param {*} dataRaw
   */
  private getDataOneHot(dataRaw: RawData): Metadata {
    const meta = Object.assign({}, this.meta);

    const inputMeta = this.getInputMetaOneHot(dataRaw, meta.inputs, 'xs');
    const outputMeta = this.getInputMetaOneHot(dataRaw, meta.outputs, 'ys');

    meta.inputs = inputMeta;
    meta.outputs = outputMeta;

    this.meta = {
      ...this.meta,
      ...meta,
    };

    return meta;
  }

  /**
   * getOneHotMeta
   * @param {*} inputsMeta
   * @param {*} dataRaw
   * @param {*} xsOrYs
   */
  private getInputMetaOneHot(dataRaw: RawData, inputsMeta: KeyedInputMeta, xsOrYs: 'xs' | 'ys'): KeyedInputMeta {
    return Object.entries(inputsMeta).reduce((meta, [key, {dtype}]) => {
      if (dtype === 'string') {
        const uniqueVals = [...new Set(dataRaw.map(obj => obj[xsOrYs][key]))];
        const oneHotMeta = this.createOneHotEncodings(uniqueVals as string[]);
        return {
          ...meta,
          [key]: {
            ...meta[key],
            ...oneHotMeta
          }
        }
      } else return meta;
    }, inputsMeta);
  }

  /**
   * Returns a legend mapping the
   * data values to oneHot encoded values
   */
  // eslint-disable-next-line class-methods-use-this, no-unused-vars
  private createOneHotEncodings(uniqueValuesArray: string[]): {legend: Legend; uniqueValues: string[]} {
    return tf.tidy(() => {
      const uniqueValues = uniqueValuesArray; // [...new Set(this.data.raw.map(obj => obj.xs[prop]))]
      // get back values from 0 to the length of the uniqueVals array
      const onehotValues = uniqueValues.map((_, idx) => idx);
      // oneHot encode the values in the 1d tensor
      const oneHotEncodedValues = tf.oneHot(tf.tensor1d(onehotValues, 'int32'), uniqueValues.length);
      // convert them from tensors back out to an array
      const oneHotEncodedValuesArray = oneHotEncodedValues.arraySync() as number[][];

      // populate the legend with the key/values
      const legend = Object.fromEntries(
          uniqueValues.map((uVal, uIdx) => [uVal, oneHotEncodedValuesArray[uIdx]])
      );
      return {
        legend,
        uniqueValues
      };
    });
  }

  /**
   * ////////////////////////////////////////////////
   * saving / loading data
   * ////////////////////////////////////////////////
   */

  /**
   * Loads data from a URL using the appropriate function
   * @param {*} dataUrl
   * @param {*} inputs
   * @param {*} outputs
   */
  public async loadDataFromUrl(dataUrl: string, inputs: Label[], outputs: Label[]) {
    try {
      let result;

      if (dataUrl.endsWith('.csv')) {
        result = await this.loadCSV(dataUrl, inputs, outputs);
      } else if (dataUrl.endsWith('.json')) {
        result = await this.loadJSON(dataUrl, inputs, outputs);
      } else if (dataUrl.includes('blob')) {
        result = await this.loadBlob(dataUrl, inputs, outputs);
      } else {
        throw new Error('Not a valid data format. Must be csv or json');
      }

      return result;
    } catch (error) {
      console.error(error);
      throw new Error(error);
    }
  }

  /**
   * loadJSON
   * @param {*} dataUrlOrJson
   * @param {*} inputLabels
   * @param {*} outputLabels
   */
  private async loadJSON(dataUrlOrJson, inputLabels: Label[], outputLabels: Label[]) {
    try {
      let json;
      // handle loading parsedJson
      if (dataUrlOrJson instanceof Object) {
        json = Object.assign({}, dataUrlOrJson);
      } else {
        const {data} = await axios.get(dataUrlOrJson);
        json = data;
      }

      // format the data.raw array
      return this.formatRawData(json, inputLabels, outputLabels);
    } catch (err) {
      console.error('error loading json');
      throw new Error(err);
    }
  }

  /**
   * loadCSV
   * @param {*} dataUrl
   * @param {*} inputLabels
   * @param {*} outputLabels
   */
  private async loadCSV(dataUrl: string, inputLabels: Label[], outputLabels: Label[]) {
    try {
      const myCsv = tf.data.csv(dataUrl);
      const loadedData = await myCsv.toArray();
      const json = {
        entries: loadedData,
      };
      // format the data.raw array
      return this.formatRawData(json, inputLabels, outputLabels);
    } catch (err) {
      console.error('error loading csv', err);
      throw new Error(err);
    }
  }

  /**
   * loadBlob
   * @param {*} dataUrlOrJson
   * @param {*} inputLabels
   * @param {*} outputLabels
   */
  private async loadBlob(dataUrlOrJson, inputLabels: Label[], outputLabels: Label[]) {
    try {
      const {data} = await axios.get(dataUrlOrJson);
      const text = data; // await data.text();

      let result;
      if (nnUtils.isValidJson(text)) {
        const json = JSON.parse(text);
        result = await this.loadJSON(json, inputLabels, outputLabels);
      } else {
        const json = this.csvToJSON(text);
        result = await this.loadJSON(json, inputLabels, outputLabels);
      }

      return result;
    } catch (err) {
      console.log('mmm might be passing in a string or something!', err);
      throw new Error(err);
    }
  }

  /**
   * loadData from fileinput or path
   * @param {*} filesOrPath
   */
  public async loadData(filesOrPath: string | FileList): Promise<void> {
    let loadedData;

    if (typeof filesOrPath === 'string') {
      const loadedData = await loadFile(filesOrPath);
    } else {
      const file = filesOrPath[0];
      if (!file) {
        throw new Error("No files found in FileList");
      }
      const text = await file.text();
      loadedData = JSON.parse(text);
    }

      this.data.raw = this.findEntries(loadedData);

      // check if a data or entries property exists
      if (!this.data.raw.length > 0) {
        console.log('data must be a json object containing an array called "data" or "entries');
      }
  }

  /**
   * saveData
   * @param {*} name
   */
  public async saveData(name?: string): Promise<void> {
    const today = new Date();
    const date = `${String(today.getFullYear())}-${String(today.getMonth() + 1)}-${String(
      today.getDate(),
    )}`;
    const time = `${String(today.getHours())}-${String(today.getMinutes())}-${String(
      today.getSeconds(),
    )}`;
    let dataName = `${date}_${time}`;
    if (name) dataName = name;

    const output = {
      data: this.data.raw,
    };

    await saveBlob(JSON.stringify(output), `${dataName}.json`, 'text/plain');
  }

  /**
   * Saves metadata of the data
   */
  public async saveMeta(nameOrCb, cb) {
    const { string, callback } = handleArguments(nameOrCb, cb);
    const modelName = string || 'model';

    await saveBlob(JSON.stringify(this.meta), `${modelName}_meta.json`, 'text/plain');
    if (callback) {
      callback();
    }
  }

  /**
   * load a model and metadata
   * @param {*} filesOrPath
   */
  public async loadMeta(filesOrPath: string | FileList | {model: string; weights: string; metadata: string}): Promise<void> {
    if ( typeof filesOrPath === "string" ) {
      const metaPath = `${filesOrPath.substring(0, filesOrPath.lastIndexOf('/'))}/model_meta.json`;
      this.meta = await loadFile(metaPath);
    }
    else if (filesOrPath instanceof FileList) {
      // TODO: what happens to the other files?
      const files = await Promise.all(
        Array.from(filesOrPath).map(async file => {
          if (file.name.includes('.json') && !file.name.includes('_meta')) {
            return {
              name: 'model',
              file,
            };
          } else if (file.name.includes('.json') && file.name.includes('_meta.json')) {
            const modelMetadata = await file.text();
            return {
              name: 'metadata',
              file: modelMetadata,
            };
          } else if (file.name.includes('.bin')) {
            return {
              name: 'weights',
              file,
            };
          }
          return {
            name: null,
            file: null,
          };
        }),
      );

      this.meta = JSON.parse(files.find(item => item.name === 'metadata').file);
    } else {
      // filesOrPath = {model: URL, metadata: URL, weights: URL}
      this.meta = await loadFile(filesOrPath.metadata);
    }

    this.isMetadataReady = true;
    this.isWarmedUp = true;

    if (callback) {
      callback();
    }
    return this.meta;
  }

  /*
   * ////////////////////////////////////////////////
   * data loading helpers
   * ////////////////////////////////////////////////
   */

  /**
   * // TODO: convert ys into strings, if the task is classification
    // if (this.config.architecture.task === "classification" && typeof output.ys[prop] !== "string") {
    //   output.ys[prop] += "";
    // }
   * formatRawData
   * takes a json and set the this.data.raw
   * @param {*} json 
   * @param {Array} inputLabels
   * @param {Array} outputLabels
   */
  private formatRawData(json, inputLabels: Label[], outputLabels: Label[]) {
    // Recurse through the json object to find
    // an array containing `entries` or `data`
    const dataArray = this.findEntries(json);

    if (dataArray.length === 0) {
      console.log(`your data must be contained in an array in \n
        a property called 'entries' or 'data' of your json object`);
    }

    // create an array of json objects [{xs,ys}]
    const result = dataArray.map((item, idx) => {
      const output: RawData[number] = {
        xs: {},
        ys: {},
      };

      inputLabels.forEach(k => {
        if (item[k] !== undefined) {
          output.xs[k] = item[k];
        } else {
          console.error(`the input label ${k} does not exist at row ${idx}`);
        }
      });

      outputLabels.forEach(k => {
        if (item[k] !== undefined) {
          output.ys[k] = item[k];
        } else {
          console.error(`the output label ${k} does not exist at row ${idx}`);
        }
      });

      return output;
    });

    // set this.data.raw
    this.data.raw = result;

    return result;
  }

  /**
   * csvToJSON
   * Creates a csv from a string
   * @param {*} csv
   */
  // TODO: where are numeric strings converted to string?
  // via: http://techslides.com/convert-csv-to-json-in-javascript
  // eslint-disable-next-line class-methods-use-this
  private csvToJSON(csv: string): {entries: Record<string, string>[]} {
    // split the string by linebreak
    const lines = csv.split('\n');
    const result = [];
    // get the header row as an array
    const headers = lines[0].split(',');

    // iterate through every row
    for (let i = 1; i < lines.length; i += 1) {
      // create a json object for each row
      const row: Record<string, any> = {};
      // split the current line into an array
      const currentline = lines[i].split(',');

      // for each header, create a key/value pair
      headers.forEach((k, idx) => {
        row[k] = currentline[idx];
      });
      // add this to the result array
      result.push(row);
    }

    return {
      entries: result,
    };
  }

  /**
   * findEntries
   * recursively attempt to find the entries
   * or data array for the given json object
   * @param {*} data
   */
  private findEntries(data: Record<string, any>): Record<string, string | number>[] {
    const parentCopy = Object.assign({}, data);

    if (parentCopy.entries && parentCopy.entries instanceof Array) {
      return parentCopy.entries;
    } else if (parentCopy.data && parentCopy.data instanceof Array) {
      return parentCopy.data;
    }

    const keys = Object.keys(parentCopy);
    // eslint-disable-next-line consistent-return
    keys.forEach(k => {
      if (typeof parentCopy[k] === 'object') {
        return this.findEntries(parentCopy[k]);
      }
    });

    return parentCopy;
  }

  /**
   * getData
   * return data object's raw array
   * to make getting raw data easier
   */
  getData() {
    const rawArray = this.data.raw;
    return rawArray;
  }
}

export default NeuralNetworkData;
