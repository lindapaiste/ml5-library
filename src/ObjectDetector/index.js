// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
  ObjectDetection
*/

import { YOLO } from "./YOLO/index";
import {CocoSsd} from "./CocoSsd/index";
import {ArgSeparator} from "../utils/argSeparator";

class ObjectDetector {
  /**
   * @typedef {Object} ObjectDetectorOptions
   * @property {number} filterBoxesThreshold - Optional. default 0.01
   * @property {number} IOUThreshold - Optional. default 0.4
   * @property {number} classProbThreshold - Optional. default 0.4
   */
  /**
   * Create ObjectDetector model. Works on video and images.
   * @param {string} modelNameOrUrl - The name or the URL of the model to use. Current model name options
   *    are: 'YOLO' and 'CocoSsd'.
   * @param {HTMLVideoElement} [video]
   * @param {ObjectDetectorOptions} [options] - Optional. A set of options.
   * @param {function} [callback] - Optional. A callback function that is called once the model has loaded.
   */
  constructor(modelNameOrUrl, video, options, callback) {
    this.video = video;
    this.modelNameOrUrl = modelNameOrUrl;
    this.options = options || {};
    this.callback = callback;

    switch (modelNameOrUrl) {
      case "yolo":
        this.model = new YOLO(
          this.video,
          {
            disableDeprecationNotice: true,
            ...this.options,
          },
          callback,
        );
        return this;
      case "cocossd":
        this.model = new CocoSsd(this.video, this.options, callback);
        return this;
      default:
        // use cocossd as default
        this.model = new CocoSsd(this.video, this.options, callback);
        return this;
    }
  }
}

const objectDetector = (modelName, videoOrOptionsOrCallback, optionsOrCallback, cb) => {
  const {string, video, options, callback} = new ArgSeparator(modelName, videoOrOptionsOrCallback, optionsOrCallback, cb);
  /* don't need this to be an error because there is a default of CocoSsd
   if ( ! string ) {
    throw new Error('Please specify a model to use. E.g: "YOLO"');
  } */
  const instance = new ObjectDetector(string, video, options, callback);

  return instance.model.callback ? instance.model : instance.model.ready;
};

export default objectDetector;
