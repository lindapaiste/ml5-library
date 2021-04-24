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
 * @template T - the type of the resolved Promise
 * @param {Promise<T>} promise
 * @param {Callback<T>} callback
 * @return {Promise<any>}
 */
export default function callCallback<T>(promise: Promise<T>, callback: Callback<T>): Promise<T | any> {
    if (callback) {
        promise
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
