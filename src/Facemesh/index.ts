// Copyright (c) 2020 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
 * Facemesh: Facial landmark detection in the browser
 * Ported and integrated from all the hard work by: https://github.com/tensorflow/tfjs-models/tree/master/facemesh
 */

import * as tf from "@tensorflow/tfjs";
import * as facemeshCore from "@tensorflow-models/facemesh";
import { EventEmitter } from "events";
import callCallback, {Callback} from "../utils/callcallback";
import {ArgSeparator} from "../utils/argSeparator";
import {TfImageSource} from "../utils/imageUtilities";
import {AbstractImageVideoModel} from "../utils/model-composition/ModelWrapper";
import {createFactory} from "../utils/model-composition/AsyncModel";

interface FacemeshOptions {
  maxContinuousChecks?: number;
  detectionConfidence?: number;
  maxFaces?: number;
  iouThreshold?: number;
  scoreThreshold?: number;
  flipHorizontal?: boolean;
  returnTensors?: boolean;
}

class Facemesh extends EventEmitter {
  ready: Promise<Facemesh>;
  modelReady: boolean;
  config: FacemeshOptions;
  video?: HTMLVideoElement;
  model?: facemeshCore.FaceMesh;

  /**
   * Create Facemesh.
   * @param {HTMLVideoElement} [video] - An HTMLVideoElement.
   * @param {object} [options] - An object with options.
   * @param {function} [callback] - A callback to be called when the model is ready.
   */
  constructor(video?: HTMLVideoElement, options: FacemeshOptions = {}, callback?: Callback<Facemesh>) {
    super();

    this.video = video;
    this.modelReady = false;
    this.config = options;

    this.ready = callCallback(this.loadModel(), callback);
  }

  /**
   * Load the model and set it to this.model
   * @return {this} the Facemesh model.
   */
  async loadModel() {
    this.model = await facemeshCore.load(this.config);
    this.modelReady = true;

    if (this.video && this.video.readyState === 0) {
      await new Promise<void>(resolve => {
        this.video!.onloadeddata = () => {
          resolve();
        };
      });
    }

    // TODO why is this needed?
    await this.predict();

    return this;
  }

  /**
   * Load the model and set it to this.model
   * @param {ImageArg | function} inputOr
   * @param {function} [cb]
   * @return {this} the Facemesh model.
   */
  async predict(inputOr?: TfImageSource | Callback<any>, cb?: Callback<any>): Promise<facemeshCore.AnnotatedPrediction[]> {
    const {image, callback} = new ArgSeparator(this.video, inputOr, cb);
    if ( ! image ) {
      throw new Error("No image provided");
    }
    const { flipHorizontal } = this.config;
    await this.ready;
    const result = await this.model!.estimateFaces(image, flipHorizontal);
    this.emit("predict", result);

    // TODO: is this right?
    if (this.video) {
      return tf.nextFrame().then(() => this.predict());
    }

    if (typeof callback === "function") {
      callback(result);
    }

    return result;
  }
}

class FacemeshNew extends AbstractImageVideoModel<facemeshCore.FaceMesh, FacemeshOptions> {

  async loadModel() {
    return await facemeshCore.load(this.config);
  }

  defaultConfig(): FacemeshOptions {
    return {};
  }

  private async predictInternal(model: facemeshCore.FaceMesh, image: TfImageSource): Promise<facemeshCore.AnnotatedPrediction[]> {
    const { flipHorizontal } = this.config;
    return await model.estimateFaces(image, flipHorizontal);
  }

  // TODO: where to handle chaining for video?
  // if (this.video) { return tf.nextFrame().then(() => this.predict()); }

  public predict = this._makeImageMethod(this.predictInternal, "predict");
}

const ff = createFactory(FacemeshNew);

const i = new FacemeshNew();

i.predict(new Image(), console.log);
i.predict(console.log, new Image());

const facemesh = (videoOrOptionsOrCallback?: HTMLVideoElement | FacemeshOptions | Callback<Facemesh>, optionsOrCallback?: FacemeshOptions | Callback<Facemesh>, cb?: Callback<Facemesh>) => {
  const {video, options, callback} = new ArgSeparator(videoOrOptionsOrCallback, optionsOrCallback, cb);
  const instance = new Facemesh(video, options, callback);
  return callback ? instance : instance.ready;
};

export default facemesh;
