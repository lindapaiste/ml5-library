// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
  ObjectDetection
*/

import {YOLOBase, YoloOptions} from "./YOLO";
import {CocoSsdBase, CocoSsdOptions} from "./CocoSsd";
import {ArgSeparator, BasicArgs} from "../utils/argSeparator";
import {ImageArg, TfImageSource, VideoArg} from "../utils/imageUtilities";
import {Callback} from "../utils/callcallback";
import {LabelAndConfidence} from "../utils/gettopkclasses";
import {AbstractImageVideoModel} from "../utils/ModelWrapper";


export type ObjectDetectorOptions = YoloOptions & CocoSsdOptions;

export interface Rectangle {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Includes:
 * x, y, width, height of the prediction box in pixels.
 * label of the prediction and confidence score from 0 to 1.
 */
export interface ObjectDetectorPrediction extends Rectangle, LabelAndConfidence {
  /**
   * the prediction box normalized from 0 to 1
   */
  normalized: Rectangle;
}

/**
 * A model is a valid ObjectDetector if it has a detect() method that takes an image
 * and returns an array of ObjectDetectorPrediction
 */
export interface ObjectDetector {
  detect(image: TfImageSource): Promise<ObjectDetectorPrediction[]>;
  ready: Promise<ObjectDetector>;
}

class ObjectDetectorWrapper extends AbstractImageVideoModel<ObjectDetector, ObjectDetectorOptions> {

  // TODO
  isPredicting: boolean = false;

  private modelName?: string;

  constructor(args: BasicArgs & {options?: ObjectDetectorOptions, callback?: Callback<ObjectDetector>}) {
    super(args);

    this.modelName = args.string;
  }

  defaultConfig(): ObjectDetectorOptions {
    return {};
  }

  /**
   * Create an instance of YOLO or CocoSsd
   */
  private instantiateDetector(): ObjectDetector {
    switch (this.modelName?.toLowerCase()) {
      case "yolo":
        return new YOLOBase(
            {
              disableDeprecationNotice: true,
              ...this.config,
            }
        );
      case "cocossd":
      default:
        // use cocossd as default
        return new CocoSsdBase(this.config);
    }
  }

  /**
   * Waits for the model to be ready before setting this.model
   */
  loadModel(): Promise<ObjectDetector> {
    const detector = this.instantiateDetector();
    return detector.ready;
  }

  /**
   * Async function which returns an array of ObjectDetectorPrediction
   */
  public detect = this._makeImageMethod((model, image) => model.detect(image), "detect");

}

/**
 * Create ObjectDetector model. Works on video and images.
 * @param {string} modelNameOrUrl - The name or the URL of the model to use. Current model name options
 *    are: 'YOLO' and 'CocoSsd'.
 * @param {HTMLVideoElement} [video]
 * @param {ObjectDetectorOptions} [options] - Optional. A set of options.
 * @param {function} [callback] - Optional. A callback function that is called once the model has loaded.
 */
const objectDetector = (modelNameOrUrl: string, video?: VideoArg | ObjectDetectorOptions | Callback<ObjectDetector>, options?: ObjectDetectorOptions | Callback<ObjectDetector>, callback?: Callback<ObjectDetector>) => {
  const args = new ArgSeparator(modelNameOrUrl, video, options, callback);
  /* don't need this to be an error because there is a default of CocoSsd
   if ( ! string ) {
    throw new Error('Please specify a model to use. E.g: "YOLO"');
  } */

  let instance: ObjectDetector;

  switch (args.string?.toLowerCase()) {
    case "yolo":
      instance = new YOLO(
          args.video,
          {
            disableDeprecationNotice: true,
            ...args.options,
          },
          args.callback,
      );
      break;
    case "cocossd":
    default:
      // use cocossd as default
      instance = new CocoSsd(args.video, args.options, args.callback);
  }

  return args.callback ? instance : instance.ready;
};

type BaseArgs = Parameters<typeof objectDetector> extends [any, ...(infer T)] ? T : never;

export const specificObjectDetector = (modelName: string) => (...args: BaseArgs) => objectDetector(modelName, ...args);

export default objectDetector;
