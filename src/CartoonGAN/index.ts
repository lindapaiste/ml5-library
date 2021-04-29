// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
/*
* CartoonGAN: see details about the [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)
*/

import * as tf from '@tensorflow/tfjs';
import callCallback, {Callback} from '../utils/callcallback';
import {GeneratedImageResult, generatedImageResult} from "../utils/GeneratedImage";
import {ArgSeparator} from "../utils/argSeparator";
import {ImageArg, TfImageSource} from "../utils/imageUtilities";
import {toTensor} from "../utils/imageConversion";

const IMAGE_SIZE = 256;

const SUPPORTED_MODELS = {
  'hosoda': 'https://raw.githubusercontent.com/leemengtaiwan/tfjs-models/master/cartoongan/tfjs_json_models/hosoda/model.json',
  'miyazaki': 'https://raw.githubusercontent.com/Derek-Wds/training_CartoonGAN/master/tfModels/Miyazaki/model.json'
};

interface CartoonOptions {
  modelUrl?: string;
  returnTensors?: boolean;
  // image size?
}

class Cartoon {
  ready: Promise<Cartoon>;
  config: Required<CartoonOptions>;
  model?: tf.GraphModel;

  /**
     * Create a CartoonGan model.
     * @param {CartoonOptions} options - Required. The name of pre-included model or the url path to your model.
     * @param {function} callback - Required. A function to run once the model has been loaded.
     */
  constructor(options: CartoonOptions = {}, callback?: Callback<Cartoon>) {
    this.config = {
      // TODO: lookup name against supported models
      modelUrl: options.modelUrl ? options.modelUrl : SUPPORTED_MODELS.miyazaki,
      returnTensors: options.returnTensors ? options.returnTensors : false,
    }
    this.ready = callCallback(this.loadModel(this.config.modelUrl), callback);
  }

  /* load tfjs model that is converted by tensorflowjs with graph and weights */
  async loadModel(modelUrl: string) {
    this.model = await tf.loadGraphModel(modelUrl);
    return this;
  }

  /**
     * generate an img based on input Image.
     * @param {HTMLImageElement | HTMLCanvasElement} inputOrCallback the source img you want to transfer.
     * @param {function} cb
     */
  async generate(inputOrCallback: ImageArg | Callback<GeneratedImageResult>, cb: Callback<GeneratedImageResult>) {

    const {image, callback} = ArgSeparator.from(inputOrCallback, cb)
        .require(
            'image',
            'Detection subject not supported'
        );

    return callCallback(this.generateInternal(image), callback);
  }

  async generateInternal(src: TfImageSource) {
    await this.ready;
    await tf.nextFrame();
    // adds resizeBilinear to resize image to 256x256 as required by the model
    const img = toTensor(src).resizeBilinear([IMAGE_SIZE,IMAGE_SIZE]);
    if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
      throw new Error(`Input size should be ${IMAGE_SIZE}*${IMAGE_SIZE} but ${img.shape} is found`);
    } else if (img.shape[2] !== 3) {
      throw new Error(`Input color channel number should be 3 but ${img.shape[2]} is found`);
    }
    const t4d = img.sub(127.5).div(127.5).reshape<tf.Rank.R4>([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        
    const alpha = tf.ones([IMAGE_SIZE, IMAGE_SIZE, 1]).tile([1, 1, 1]).mul<tf.Tensor3D>(255)
    const res = (this.model!.predict(t4d) as tf.Tensor).add(1).mul(127.5)
        .reshape<tf.Rank.R3>([IMAGE_SIZE, IMAGE_SIZE, 3]).floor().concat<tf.Tensor3D>(alpha, 2)
    const result = await generatedImageResult(res, this.config);
        
    if(this.config.returnTensors){
      return result;
    }

    // TODO: some should be disposed regardless
    img.dispose();
    res.dispose();
    t4d.dispose();
    alpha.dispose();
    return result;
  }
} 

const cartoon = (optionsOr: CartoonOptions | string | Callback<Cartoon>, cb?: Callback<Cartoon>) => {
  const {string, options = {}, callback} = new ArgSeparator(optionsOr, cb);
  if ( string ) {
    options.modelUrl = string;
  }
  return new Cartoon(options, callback);
};


export default cartoon; 
