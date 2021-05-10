import {DType} from "./NeuralNetworkData";

export default {
    /**
     * normalizeValue
     * @param {*} value
     * @param {*} min
     * @param {*} max
     */
    normalizeValue: (value: number, min: number, max: number): number => {
        return ((value - min) / (max - min))
    },

    /**
     * unNormalizeValue
     * @param {*} value
     * @param {*} min
     * @param {*} max
     */
    unnormalizeValue(value: number, min: number, max: number): number {
        return ((value * (max - min)) + min)
    },

    /**
     * arrayMin
     * @param {*} array
     */
    arrayMin(array: number[]): number {
        // return Math.min(..._array)
        return array.reduce((a, b) => {
            return Math.min(a, b);
        });
    },

    /**
     * arrayMax
     * @param {*} array
     */
    arrayMax(array: number[]): number {
        return array.reduce((a, b) => {
            return Math.max(a, b);
        });
        // return Math.max(..._array)
    },

    /**
     * checks whether or not a string is a json
     * @param {*} str
     */
    isValidJson(str: string): boolean {
        try {
            JSON.parse(str);
        } catch (e) {
            return false;
        }
        return true;
    },

    /**
     * zipArrays
     * @param {*} arr1
     * @param {*} arr2
     */
    zipArrays<A, B>(arr1: A[], arr2: B[]): (A & B)[] {
        if (arr1.length !== arr2.length) {
            throw new Error('arrays do not have the same length');
        }

        return [...new Array(arr1.length)].map((item, idx) => {
            return {
                ...arr1[idx],
                ...arr2[idx]
            }
        });
    },

    /**
     * createLabelsFromArrayValues
     * @param {*} incoming
     * @param {*} prefix
     */
    createLabelsFromArrayValues(incoming: any[], prefix: string): string[] {
        return incoming.map((_, idx) => `${prefix}_${idx}`);
    },

    /**
     * takes an array and turns it into a json object
     * where the labels are the keys and the array values
     * are the object values
     * @param {*} incoming
     * @param {*} labels
     */
    formatDataAsObject<T>(incoming: T[] | Record<any, T>, labels: string[]): Record<string, T> {
        if (typeof incoming !== 'object') {
            throw new Error('input provided is not supported or does not match your output label specifications');
        }
        // TODO could use object.values to apply labels to object inputs, but is this desired?
        // previously just returned the object
        return Object.values(incoming).reduce((obj, item, idx) => ({
            ...obj,
            [labels[idx]]: item
        }), {});
    },

    /**
     * returns a datatype of the value as string
     * @param {*} val
     */
    getDataType(val: any): DType {
        const dtype = typeof val;

        if (dtype === 'object' && Array.isArray(val)) {
            return DType.ARRAY;
        }
        if ( this.isDataType(dtype) ) {
            return dtype;
        }
        throw new Error(`Invalid data type ${dtype}. Data type must be one of: ${Object.values(DType).join(', ')}`)
    },

    /**
     * validate that a string is one of the acceptable data types
     * @param dtype
     */
    isDataType(dtype: string): dtype is DType {
        return (Object.values(DType) as string[]).includes(dtype);
    },
}