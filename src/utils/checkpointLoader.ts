// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tf from '@tensorflow/tfjs';
import {loadFile} from "./io";
import modelLoader, {ModelLoader} from "./modelLoader";

/**
 * CheckpointLoader handles a manifest.json file containing the file names and tensor shapes for other variables.
 * It loads the manifest and then loads all related files, constructing the correct tensors based on the shape.
 */

type ManifestType = Record<string, { shape: number[]; filename: string; }>

export default class CheckpointLoader {
    private readonly url: ModelLoader;
    private checkpointManifest?: ManifestType;
    private variables: Record<string, tf.Tensor> = {};
    private didLoadVariables: boolean = false;

    /**
     * @param urlPath - expected path is the directory which contains the manifest.json,
     * but the ModelLoader helper allows for some leniency. The path to a file in that directory will work too.
     */
    constructor(urlPath: string) {
        this.url = modelLoader(urlPath);
    }

    /**
     * Loads or retrieves the manifest file, which contains the specs for each variable.
     */
    async getCheckpointManifest(): Promise<ManifestType> {
        if (this.checkpointManifest === undefined) {
            this.checkpointManifest = await loadFile<ManifestType>(
                this.url.fileInDirectory("manifest.json")
            );
        }
        return this.checkpointManifest;
    }

    /**
     * Returns a dictionary of all variables keyed by their names.
     * Loads if not already loaded.
     */
    async getAllVariables(): Promise<Record<string | number, tf.Tensor>> {
        if (this.didLoadVariables) {
            return this.variables;
        }
        const manifest = await this.getCheckpointManifest();
        const variableNames = Object.keys(manifest);
        const variablePromises = variableNames.map(v => this.getVariable(v));
        // individual getVariable calls will set this.variables, so just return it
        await Promise.all(variablePromises);
        this.didLoadVariables = true;
        return this.variables;
    }

    /**
     * Loads a variable from the filename in the Manifest, or returns the local variable if already loaded.
     * Saves to this.variables to prevent duplicated requests.
     * @param varName
     */
    async getVariable(varName: string): Promise<tf.Tensor> {
        // access the manifest
        const manifest = await this.getCheckpointManifest();
        // don't need to fetch again if already loaded
        if (varName in this.variables) {
            return this.variables[varName];
        }
        if (!(varName in manifest)) {
            throw new Error(`Cannot load non-existent variable ${varName}`);
        }

        try {
            // load the ArrayBuffer
            const buffer = await loadFile(
                this.url.fileInDirectory(manifest[varName].filename),
                "arraybuffer"
            );
            // get the array of numbers
            const values = new Float32Array(buffer);
            // convert to a tensor
            const tensor = tf.tensor(values, manifest[varName].shape);
            // save the variable for future access
            this.variables[varName] = tensor;
            // return the tensor
            return tensor;
        } catch (error) {
            throw new Error(`Could not fetch variable ${varName}: ${error?.message}`);
        }
    }
}