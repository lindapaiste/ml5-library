import * as USE from '@tensorflow-models/universal-sentence-encoder';
import callCallback, {Callback} from '../utils/callcallback';
import {AsyncModel, constructWithCallback, Ready} from "../utils/model-composition/AsyncModel";
import {ArgSeparator} from "../utils/argSeparator";


/**
 * @typedef {Object} USEOptions
 * @property {boolean} withTokenizer
 */
interface USEOptions {
  withTokenizer?: boolean;
}

/**
 * @property {(USE.UniversalSentenceEncoder | null)} model
 * @property {(USE.Tokenizer | null)} tokenizer
 * @property {USEOptions} config
 */
class UniversalSentenceEncoder extends AsyncModel<USE.UniversalSentenceEncoder, USEOptions> {
  tokenizer?: USE.Tokenizer;

  /**
   * @private
   * load model
   */
  async loadModel(){
    if(this.config.withTokenizer){
      this.tokenizer = await USE.loadTokenizer();
    }
    return USE.load();
  }

  defaultConfig(): USEOptions {
    return {withTokenizer: false}
  }

  /**
   * @public
   * @param {(string | string[])} text
   * @param {function} callback
   * @return {Promise}
   */
  predict(text: string | string[], callback?: Callback<number[][]>){
    return this.callWhenReady(this.predictInternal, callback, "predict", text);
  }

  /**
   * @private
   * @param {(string | string[])} text
   * @return {Promise<number[][]>}
   */
  async predictInternal(this: Ready<this>, text: string | string[]){
      const embeddings = await this.model.embed(text);
      const results = await embeddings.array();
      embeddings.dispose();
      return results;
  }

  /**
   * Encodes a string based on the loaded tokenizer if the withTokenizer:true
   * @public
   * @param {string} textString
   * @param {function} callback
   * @return {Promise}
   */
  encode(textString: string, callback?: Callback<number[]>){
    return this.callWhenReady(this.encodeInternal, callback, "encode", textString);
  }

  /**
   * @private
   * @param {string} textString
   * @return {Promise<boolean|Uint8Array>}
   */
  async encodeInternal(textString: string){
    if(this.config.withTokenizer && this.tokenizer){
      return this.tokenizer.encode(textString);
    }
    throw new Error('withTokenizer must be set to true - please pass "withTokenizer:true" as an option in the constructor');
  }

}

/**
 * @param {function | Partial<USEOptions>} [optionsOr]
 * @param {function} [cb]
 * @return {UniversalSentenceEncoder | Promise<UniversalSentenceEncoder>}
 */
const universalSentenceEncoder = (optionsOr?: USEOptions | Callback<UniversalSentenceEncoder>, cb?: Callback<UniversalSentenceEncoder>) => {
  const {options, callback} = new ArgSeparator(optionsOr, cb);
  return constructWithCallback(UniversalSentenceEncoder, options, callback);
};

export default universalSentenceEncoder;