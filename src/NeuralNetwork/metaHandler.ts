import nnUtils from "./NeuralNetworkUtils";
import * as tf from "@tensorflow/tfjs";
import {
    DType,
    KeyedInputMeta,
    InputMeta,
    Legend,
    Metadata,
    RawData,
    RecursiveRawPropertyData,
    XsOrYs
} from "./NeuralNetworkData";
import {ObjectBuilder} from "../utils/objectUtilities";
import {Tensor} from "@tensorflow/tfjs-core";
import {ImageArg} from "../utils/imageUtilities";
import {isSameArray} from "../utils/arrayUtilities";

/**
 * Normalization from raw values (string, numbers, arrays) to numbers.
 * TODO: boolean?
 *
 * All of the information about the normalization task needs to be stored in a `meta` object
 * so that the correct handler instance can be re-created based on data in a saved JSON file.
 */

// Don't know how to make an abstract base class without using generics
interface PropertyHandlerInstance<In, Out> {
    dtype: DType;
    min: number;
    max: number;

    getMetadata(): InputMeta;

    getUnits(): number;

    getShape(): number[];

    normalizeValue(value: In): Out;

    unnormalizeValue(value: Out): In;

    canAddValue?(value: In): boolean;
}

interface PropertyHandlerClass<In, Out> {
    fromMeta(meta: InputMeta): PropertyHandlerInstance<In, Out>;

    fromData(data: In[]): PropertyHandlerInstance<In, Out>;
}

/**
 * Strings are treated as a closed set of defined enums.
 * Each value is encoded into a number array using "One Hot" encoding.
 * Encoded values are an an array with all zeros and one one ie. [0,0,0,1,0,0].
 * The position of the 1 dictates which string value it is. Thus, the length of the array equals the number or options.
 */
export class StringNormalizer {
    public readonly dtype = DType.STRING;
    public readonly min = 0;
    public readonly max = 1;

    constructor(private readonly legend: Legend, private readonly uniqueValues: string[]) {
    }

    public static fromMeta(meta: InputMeta): StringNormalizer {
        const {legend, uniqueValues} = meta;
        if (!legend || !uniqueValues || meta.dtype !== DType.STRING) {
            throw new Error(`Invalid meta. Can only construct a StringNormalizer if meta has dtype ${DType.STRING} and includes properties legend and uniqueValue`);
        }
        return new StringNormalizer(legend, uniqueValues);
    }

    public static fromData(raw: string[]): StringNormalizer {
        const uniqueValues = [...new Set(raw)];
        const oneHotEncodedValuesArray = tf.tidy(() => {
            // get back values from 0 to the length of the uniqueVals array
            const onehotValues = uniqueValues.map((_, idx) => idx);
            // oneHot encode the values in the 1d tensor
            const oneHotEncodedValues = tf.oneHot(tf.tensor1d(onehotValues, 'int32'), uniqueValues.length);
            // convert them from tensors back out to an array
            return oneHotEncodedValues.arraySync() as number[][];
        });
        // populate the legend with the key/values
        const legend = Object.fromEntries(
            uniqueValues.map((uVal, uIdx) => [uVal, oneHotEncodedValuesArray[uIdx]])
        );
        return new StringNormalizer(legend, uniqueValues);
    }

    public getMetadata(): InputMeta {
        return {
            min: this.min,
            max: this.max,
            dtype: this.dtype,
            legend: this.legend,
            uniqueValues: this.uniqueValues,
        }
    }

    /**
     * Units is the length of the oneHot array which is the count of unique values
     */
    public getUnits(): number {
        return this.uniqueValues.length;
    }

    public getShape(): number[] {
        return [this.uniqueValues.length];
    }

    /**
     * Lookup a string and return the oneHot-encoded value
     * @param value
     */
    public normalizeValue = (value: string): number[] => {
        return this.legend[value];
    }

    /**
     * Lookup the label which corresponds to a oneHot-encoded array
     * @param value
     */
    public unnormalizeValue = (value: number[]): string => {
        // find the entry in the legend that matches the value
        const index = Object.values(this.legend).findIndex(encoding =>
            value.every((num, i) => encoding[i] === num)
        );
        if (index === -1) {
            throw new Error("No entry found in legend matching input array.");
        }
        return Object.keys(this.legend)[index];
    }

    public canAddValue(value: string): boolean {
        return value in this.legend;
    }
}

/**
 * Normalizes values from 0 to 1.
 */
class NumberNormalizer {
    public readonly dtype = DType.NUMBER;

    constructor(public readonly min: number, public readonly max: number) {
    }

    public static fromMeta(meta: InputMeta): NumberNormalizer {
        const {min, max} = meta;
        return new NumberNormalizer(min, max);
    }

    public static fromData(raw: number[]): NumberNormalizer {
        const min = nnUtils.arrayMin(raw);
        const max = nnUtils.arrayMax(raw);
        return new NumberNormalizer(min, max);
    }

    public getMetadata(): InputMeta {
        return {
            min: this.min,
            max: this.max,
            dtype: this.dtype,
        }
    }

    public getUnits(): number {
        return 1;
    }

    public getShape(): number[] {
        return [1];
    }

    /**
     * Takes a raw number value and normalizes from 0 to 1 based on the min and max
     * @param value
     */
    public normalizeValue = (value: number): number => {
        return nnUtils.normalizeValue(value, this.min, this.max);
    }

    /**
     * Takes a number from 0 to 1 and outputs that raw value based on the min and max
     * @param value
     */
    public unnormalizeValue = (value: number): number => {
        return nnUtils.unnormalizeValue(value, this.min, this.max);
    }

    public canAddValue(value: number): boolean {
        return value >= this.min && value <= this.max;
    }
}

// TODO: can it only be depth 1? or any depth?
class RecursiveArrayNormalizer {
    public readonly dtype = DType.ARRAY;
    public readonly min: number;
    public readonly max: number;

    // internally uses a separate handler for each individual element
    constructor(private handlers: PropertyNormalizer[]) {
        this.min = nnUtils.arrayMin(this.handlers.map(handler => handler.min));
        this.max = nnUtils.arrayMin(this.handlers.map(handler => handler.max));
    }

    static fromMeta(meta: InputMeta): RecursiveArrayNormalizer {
        // TODO: based on currently stored data, there is not enough information to work backwards
    }

    /**
     * Raw data is an array of arrays.  It is an array of entries where each entry is an array.
     * It's the inner array that is mapped/handled here.
     * @param raw
     */
    static fromData(raw: RecursiveRawPropertyData[][]): RecursiveArrayNormalizer {
        // get indexes based on first entry
        const handlers = Object.values(raw[0]).map(
            // data for each handler is an array with the property at that index for each entry
            (_, i) => PropertyNormalizerFactory.fromData(raw.map(entry => entry[i]))
        );
       return new RecursiveArrayNormalizer(handlers);
    }

    public getMetadata(): InputMeta {
        return {
            min: this.min,
            max: this.max,
            dtype: this.dtype,
            // what about legend / uniqueValues for nested strings?
        }
    }

    /**
     * Units should be equal to the length of the normalized array
     */
    public getUnits(): number {
        return this.handlers.reduce((sum, handler) => sum + handler.getUnits(), 0);
    }

    /**
     * Make this a static utility to use the logic outside the class - could also be an external function.
     * In order to add a dimension, all properties/elements must have the same shape.
     * Otherwise the data cannot be constructed into a multi-dimensional Tensor.
     * In that case, must flatten the tensor.
     */
    public static combineShape(handlers: {getShape(): number[]; getUnits(): number;}[]): number[] {
        if ( handlers.length === 0 ) return [];
        const elementShape = handlers[0].getShape();
        const isMultiDimensional = handlers.every(handler => isSameArray(handler.getShape(), elementShape));
        if ( isMultiDimensional ) {
            // Add one extra dimension.
            return [handlers.length, ...elementShape];
        }
        // TODO: how to flatten properly? Potentially there can be ok dimensions greater than 1.
        // Flatten to 1D where the length is the total number of units.
        const units = handlers.reduce((sum, handler) => sum + handler.getUnits(), 0);
        return [units];
    }

    public getShape(): number[] {
        return RecursiveArrayNormalizer.combineShape(this.handlers);
    }

    public normalizeValue = (value: RecursiveRawPropertyData[]): number[] => {
        // Note: need to assert type because we don't know that the handler at this index matches the value at this index
        // could also broaden types and throw errors.
        return value.map((v, i) => (this.handlers[i] as any).normalizeValue(v)).flat();
    }

    /**
     * Unnormalization needs to undo the `flat()` call in normalize.
     * Could assume that all elements have the same length and divide evenly.
     * Could work backwards based on the units from each handler.
     */
    public unnormalizeValue = (value: number[]): RecursiveRawPropertyData[] => {
        // TODO: check that this works. Expects value to be mutated by the splice.
        return this.handlers.map(handler => handler.unnormalizeValue(value.splice(handler.getUnits(), 0)));
    }

    public canAddValue(value: RecursiveRawPropertyData[]): boolean {
        return value.every((v, i) => (this.handlers[i] as any).canAddValue(v));
    }
}

class NumberArrayNormalizer {
    public readonly dtype = DType.ARRAY;

    // internally uses a NumberHandler to manage min and max normalization.
    constructor(private numberHandler: NumberNormalizer) {
    }

    static fromMeta(meta: InputMeta): NumberArrayNormalizer {
        const {min, max} = meta;
        const numberHandler = new NumberNormalizer(min, max);
        return new NumberArrayNormalizer(numberHandler);
    }

    static fromData(raw: number[][]): NumberArrayNormalizer {
        const numberHandler = NumberNormalizer.fromData(raw.flat());
        return new NumberArrayNormalizer(numberHandler);
    }

    get min() {
        return this.numberHandler.min;
    }

    get max() {
        return this.numberHandler.max;
    }

    public getMetadata(): InputMeta {
        return {
            min: this.min,
            max: this.max,
            dtype: this.dtype,
        }
    }

    public getUnits(): number {
        // TODO: this was not implemented previously and returned an empty array
        // TODO: User must input the shape of the image size correctly. <- previous note
        //return this.raw.length;
    }

    public getShape(): number[] {
        // TODO
    }

    /**
     * @param value
     */
    public normalizeValue = (value: number[]): number[] => {
        return value.map(v => this.numberHandler.normalizeValue(v));
    }

    /**
     * @param value
     */
    public unnormalizeValue = (value: number[]): number[] => {
        return value.map(v => this.numberHandler.unnormalizeValue(v));
    }

    public canAddValue(value: number[]): boolean {
        return value.every(v => this.numberHandler.canAddValue(v));
    }
}


class ImageNormalizer {
    public readonly dtype = DType.ARRAY;
    private width?: number;
    private height?: number;
    private channelCount?: number;

    constructor() {
    }

    static fromMeta(meta: InputMeta): ImageNormalizer {
    }

    static fromData(raw: (Uint8ClampedArray | ImageArg)[]): ImageNormalizer {
        // can get width and height from all types except for pixel array

    }

    get min() {

    }

    get max() {

    }

    public getMetadata(): InputMeta {
        return {
            min: this.min,
            max: this.max,
            dtype: this.dtype,
        }
    }

    public getUnits(): tf.Shape {
        return [this.width, this.height, this.channelCount];
    }

    /**
     * @param value
     */
    public normalizeValue = (value: Uint8ClampedArray | ImageArg): number[] | Uint8ClampedArray => {

    }

    /**
     * @param value
     */
    public unnormalizeValue = (value: number[]) => {

    }

    public canAddValue(value: number[]): boolean {

    }
}


type PropertyNormalizer = StringNormalizer | NumberNormalizer | NumberArrayNormalizer | RecursiveArrayNormalizer;

class PropertyNormalizerFactory {
    /**
     * Create a handler for a single property's data based on the dtype
     * @param data
     */
    public static fromData = (data: RecursiveRawPropertyData[]): PropertyNormalizer => {
        // Check if all entries have the same dtype
        const expected = nnUtils.getDataType(data[0]);
        if (!data.every(value => nnUtils.getDataType(value) === expected)) {
            throw new Error(`Inconsistent data array. Not all values in array matched the expected type ${expected}`);
        }
        // Return the correct handler for the type
        switch (expected) {
            case DType.ARRAY:
                return RecursiveArrayNormalizer.fromData(data as any);
            //return new NumberArrayHandler(data as number[][]);
            case DType.NUMBER:
                return NumberNormalizer.fromData(data as number[]);
            case DType.STRING:
                return StringNormalizer.fromData(data as string[]);
        }
    }

    /**
     * Create a handler for a single property's data based on the dtype of the saved metadata
     * @param meta
     */
    public static fromMeta = (meta: InputMeta): PropertyNormalizer => {
        // Return the correct handler for the type
        switch (meta.dtype) {
            case DType.ARRAY:
                return RecursiveArrayNormalizer.fromMeta(meta);
            case DType.NUMBER:
                return NumberNormalizer.fromMeta(meta);
            case DType.STRING:
                return StringNormalizer.fromMeta(meta);
        }
    }
}

/**
 * Standard data format is an object with multiple property names,
 * corresponding to string, number, or array values.
 */
class KeyedObjectHandler {
}

/**
 * The properties that are stored for both xs and ys
 */
interface NormalizerContent {
    keyedHandlers: Record<string, PropertyNormalizer>;
    metadata: KeyedInputMeta;
    units: number;
    shape: number[];
}

/**
 * NormalizerFromData has a tensor for each, but NormalizerFromMeta does not
 */
interface DataNormalizerContent extends NormalizerContent {
    tensor: Tensor;
}

/**
 * create the metadata from the data
 * this covers:
 *  1. getting the datatype from the data
 *  2. getting the min and max from the data
 *  3. getting the oneHot encoded values
 *  4. getting the inputShape and outputUnits from the data
 *
 * convert data to tensors
 *  1. normalize numbers
 *  2. flatten arrays
 *  3. create tensors
 *
 * Note: all logic is initiated through the constructor, so this could be a function instead of a class.
 * Methods are primarily just to break up the code for readability.
 */
class NormalizerFromData {

    public readonly xs: DataNormalizerContent;
    public readonly ys: DataNormalizerContent;

    /**
     * Create a normalizer by examining the provided training data.
     * @param rawData
     */
    constructor(private rawData: RawData) {
        // throw error on 0 length
        if (rawData.length === 0) {
            throw new Error("No data found. Add data before attempting normalization.");
        }
        this.xs = this.createContent('xs');
        this.ys = this.createContent('ys');
    }

    get meta(): Metadata {
        return {
            inputs: this.xs.metadata,
            outputs: this.ys.metadata,
            inputUnits: this.xs.shape,
            outputUnits: this.ys.units,
            isNormalized: true,
        }
    }

    get data() {
        return {
            raw: this.rawData,
            training: {
                xs: this.xs.tensor,
                ys: this.ys.tensor,
            }
        }
    }

    /**
     * Public function to normalize an input value for prediction or classification.
     * TODO: Need to pass in the labels in order to ensure that the order is correct.
     * @param xs
     */
    public normalizeInput(xs: Record<string, RecursiveRawPropertyData>): Tensor {
        return tf.tidy(() => {
                // get the normalized value for each property from its handler
                // have to assert typescript type as any because it's not known that the handler matches the property value.
                const data = Object.entries(this.xs.keyedHandlers).flatMap(
                    ([property, handler]) => (handler.normalizeValue as any)(xs[property])
                );


            return tf.tensor(data, [1, ...this.xs.shape]);
        });
    }

    /**
     * Shared logic for xs and ys. Create handlers, meta, and units. Create tensors from normalized values.
     * @param xsOrYs
     * @private
     */
    private createContent(xsOrYs: XsOrYs): DataNormalizerContent {
        // get property names from one data point -- assume that all others are the same
        const properties = Object.keys(this.rawData[0][xsOrYs]);

        // separate the raw data into a separate array of each property's data values
        const rawDataByProperty = ObjectBuilder.fromKeys(properties)
            .createValues(key => this.rawData.map(entry => entry[xsOrYs][key]));
        // create a handler for each property based on the data values
        const keyedHandlers = ObjectBuilder.from(rawDataByProperty)
            .mapValues(dataArray => PropertyNormalizerFactory.fromData(dataArray));
        // get the meta from each handler
        const metadata = ObjectBuilder.from(keyedHandlers).mapValues(handler => handler.getMetadata());
        // add up the units from each handler
        const units = Object.values(keyedHandlers).reduce((sum, handler) => sum + handler.getUnits(), 0);
        // combine the shape
        const shape = RecursiveArrayNormalizer.combineShape(Object.values(keyedHandlers));


        // create a tensor with the normalized data
        const tensor = tf.tidy(() => {
            // flattens into a 1D array, but it gets build to the correct shape by the tensor.
            // map through the row indexes
            const dataArr: number[] = this.rawData.flatMap((row, i) =>
                // get the normalized value for each property from its handler
                // have to assert typescript type as any because it's not known that the handler matches the property value.
            // TODO: make sure that the order is correct.
                Object.entries(keyedHandlers).flatMap(([property, handler]) => (handler.normalizeValue as any)(row[xsOrYs][property]))
            );

            return tf.tensor(dataArr, [this.rawData.length, units]);
        });
        return {
            keyedHandlers,
            metadata,
            units,
            shape,
            tensor
        };
    }
}

/**
 * When loading a saved model, can create a normalizer based on the saved metadata about each field.
 */
class NormalizerFromMeta {

    public readonly xs: NormalizerContent;
    public readonly ys: NormalizerContent;

    constructor(public readonly meta: Metadata) {
        const {inputs, outputs, inputUnits, outputUnits} = this.meta;
        this.xs = {
            // Inputs might be multi-dimensional
            shape: inputUnits,
            units: inputUnits.reduce((sum, n) => sum + n, 0),
            metadata: inputs,
            keyedHandlers: ObjectBuilder.from(inputs).mapValues(PropertyNormalizerFactory.fromMeta)
        }
        this.ys = {
            // Outputs are one-dimensional
            shape: [outputUnits],
            units: outputUnits,
            metadata: outputs,
            keyedHandlers: ObjectBuilder.from(outputs).mapValues(PropertyNormalizerFactory.fromMeta)
        }
    }
}