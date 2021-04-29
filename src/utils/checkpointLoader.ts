// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tf from '@tensorflow/tfjs';
import {loadFile} from "./io";
import {Tensor} from "@tensorflow/tfjs";

const MANIFEST_FILE = 'manifest.json';

type ManifestType = Record<string, { shape: number[]; filename: string; }>

export default class CheckpointLoader {
  private readonly urlPath: string;
  private checkpointManifest: null | ManifestType = null;
  private variables: Record<string, Tensor> = {};

  constructor(urlPath: string) {
    this.urlPath = urlPath;
    // enforce trailing slash
    if (! this.urlPath.endsWith('/')) {
      this.urlPath += '/';
    }
  }

  async loadManifest(): Promise<ManifestType> {
    try {
      this.checkpointManifest = await loadFile<ManifestType>(this.urlPath + MANIFEST_FILE);
      return this.checkpointManifest;
    } catch (error) {
      throw new Error(`${MANIFEST_FILE} not found at ${this.urlPath}. ${error}`);
    }
  }


  async getCheckpointManifest(): Promise<ManifestType> {
    if (this.checkpointManifest === null) {
      return this.loadManifest();
    }
    return this.checkpointManifest;
  }

  async getAllVariables() {
    if (this.variables != null) {
      return Promise.resolve(this.variables);
    }
    const manifest = await this.getCheckpointManifest();
    const variableNames = Object.keys(manifest);
    const variablePromises = variableNames.map(v => this.getVariable(v));
    // individual getVariable calls will set this.variables, so just return it
    await Promise.all(variablePromises);
    return this.variables;
  }

  /**
   * Loads a variable from the filename in the Manifest, or returns the local variable if already loaded.
   * Saves to this.variables to prevent duplicated requests.
   * @param varName
   */
  async getVariable(varName: string): Promise<Tensor> {
    const manifest = await this.getCheckpointManifest();
    // don't need to fetch again if already loaded
    if ( varName in this.variables ) {
      return this.variables[varName];
    }
    if (!(varName in manifest)) {
      throw new Error(`Cannot load non-existent variable ${varName}`);
    }
    // TODO: would be less code to use fetch or axios instead of XMLHttpRequest
    return new Promise( (resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.responseType = 'arraybuffer';
      const fname = manifest[varName].filename;
      xhr.open('GET', this.urlPath + fname);
      xhr.onload = () => {
        if (xhr.status === 404) {
          reject( new Error(`Not found variable ${varName}`) );
        }
        const values = new Float32Array(xhr.response);
        const tensor = tf.tensor(values, manifest[varName].shape);
        this.variables[varName] = tensor;
        resolve(tensor);
      };
      xhr.onerror = (error) => {
        reject (new Error(`Could not fetch variable ${varName}: ${error}`));
      };
      xhr.send();
    });
  }
}
