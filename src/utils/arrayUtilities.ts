import {isNumber, isString} from "@tensorflow/tfjs-core/dist/util";

/**
 * copied from TensorFlow rather than imported due to conflicting definitions in multiple places.
 */
export type TypedArray = Int8Array|Uint8Array|Int16Array|Uint16Array|Int32Array|
    Uint32Array|Uint8ClampedArray|Float32Array|Float64Array;

/**
 * check if a variable is any of Float32Array, Uint8ClampedArray, etc.
 * note: TensorFlow has an isTypedArray function in tf.util, but it only checks Float32Array|Int32Array|Uint8Array
 */
export const isTypedArray = (value?: any): value is TypedArray => !! value && value.byteLength !== undefined;

export const isNumberArray = (arr: any[]): arr is number[] => arr.every(isNumber);

export const isStringArray = (arr: any[]): arr is string[] => arr.every(isString);

/**
 * Shallow equality check. Checks for strict equality of each element.
 */
export const isSameArray = (a: any[], b: any): boolean => {
    return a.length === b.length && a.every((value, i) => value === b[i]);
}