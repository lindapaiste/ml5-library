// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @template T - the type of the resolved Promise
 * @callback Callback<T>
 *     @param {any} error
 *     @param {T} result
 *     @return void
 */
export type Callback<T> = (error: any, result?: T) => void;

/**
 * Helper function allows for calling a void callback function which might be undefined.
 * @param callback
 * @param args
 */
export const maybeCall = <F extends (...args: any[]) => void>(callback: F | undefined, ...args: Parameters<F>): void => {
    if (callback !== undefined) {
        callback(...args);
    }
}

/**
 * @template T - the type of the resolved Promise
 * @param {Promise<T>} promise
 * @param {Callback<T>} callback
 * @return {Promise<any>}
 */
const callCallback = async <T>(promise: Promise<T>, callback?: Callback<T>): Promise<T> => {
    if (callback) {
        return promise
            .then((result) => {
                callback(undefined, result);
                return result;
            })
            .catch((error) => {
                callback(error);
                return error;
            });
    }
    return promise;
}
/*
 try {
        const result = await promise;
        maybeCall(callback, undefined, result);
        return result;
    } catch (error) {
        maybeCall(callback, error);
        return undefined;
    }
 */

export default callCallback;