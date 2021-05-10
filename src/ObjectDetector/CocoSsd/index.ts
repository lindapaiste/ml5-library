// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
    COCO-SSD Object detection
    Wraps the coco-ssd model in tfjs to be used in ml5
*/
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import {ArgSeparator} from "../../utils/argSeparator";
import {ObjectDetector, ObjectDetectorOptions, ObjectDetectorPrediction, specificObjectDetector} from "../index";
import {TfImageSource} from "../../utils/imageUtilities";
import wrap from "../../utils/imageConversion";

export type CocoSsdOptions = cocoSsd.ModelConfig & {
  maxNumberBoxes?: number;
}

const DEFAULTS: CocoSsdOptions = {
  base: "lite_mobilenet_v2",
};

export class CocoSsdBase implements ObjectDetector {

  public config: ObjectDetectorOptions;

  public ready: Promise<ObjectDetector>;

  private model!: cocoSsd.ObjectDetection;

  /**
   * Create CocoSsd model. Works on video and images.
   * @param options
   */
  constructor(options?: ObjectDetectorOptions) {
    this.config = {
      ...DEFAULTS,
      ...options,
    };
    this.ready = this.loadModel();
  }

  /**
   * load model
   */
  async loadModel() {
    this.model = await cocoSsd.load(this.config);
    return this;
  }

  /**
   * Detect objects that are in video, returns bounding box, label, and confidence scores
   */
  async detect(subject: TfImageSource): Promise<ObjectDetectorPrediction[]> {
    const predictions = await this.model.detect(subject, this.config.maxNumberBoxes);
    return predictions.map(prediction => {
      const {class: label, score: confidence, bbox} = prediction;
      const [x, y, width, height] = bbox;
      const image = wrap(subject);
      return {
        label,
        confidence,
        x,
        y,
        width,
        height,
        normalized: {
          x: x / image.getWidth(),
          y: y / image.getHeight(),
          width: width / image.getWidth(),
          height: height / image.getHeight(),
        },
      };
    });
  }
}

/**
 * export uses ObjectDetector to wrap the CocoSsdBase class and handle callbacks
 */
export const CocoSsd = specificObjectDetector("cocossd");
