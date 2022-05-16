// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
  Crepe Pitch Detection model
  Based on https://github.com/marl/crepe/tree/gh-pages
  Original model and code: https://marl.github.io/crepe/crepe.js
*/

import * as tf from '@tensorflow/tfjs';
import callCallback, {ML5Callback} from '../utils/callcallback';
import modelLoader from "../utils/modelLoader";

class PitchDetection {
  /**
   * The pitch detection model.
   * @type {tf.LayersModel}
   * @public
   */
  model?: tf.LayersModel;
  /**
   * The AudioContext instance. Contains sampleRate, currentTime, state, baseLatency.
   * @type {AudioContext}
   * @public
   */
  audioContext: AudioContext;
  /**
   * The MediaStream instance. Contains an id and a boolean active value.
   * @type {MediaStream}
   * @public
   */
  stream: MediaStream;

  frequency: number | null;

  ready: Promise<PitchDetection>;

  /**
   * A boolean value stating whether the model instance is running or not.
   * @type {boolean}
   * @public
   */
  running: boolean = false;

  /**
   * The current pitch prediction results from the classification model.
   */
  results?: { confidence: string };


  /**
   * Create a pitchDetection.
   * @param {string} model - The path to the trained model. Only CREPE is available for now. Case insensitive.
   * @param {AudioContext} audioContext - The browser audioContext to use.
   * @param {MediaStream} stream  - The media stream to use.
   * @param {function} callback  - Optional. A callback to be called once the model has loaded. If no callback is provided, it will return a promise that will be resolved once the model has loaded.
   */
  constructor(model: string, audioContext: AudioContext, stream: MediaStream, callback?: ML5Callback<PitchDetection>) {
    this.audioContext = audioContext;
    this.stream = stream;
    this.frequency = null;
    this.ready = callCallback(this.loadModel(model), callback);
  }

  async loadModel(model: string): Promise<this> {
    // Note: previous code always added '/model.json' to provided string
    // Now allows user to provide a directory or a model file path
    // TODO: should it be made absolute?
    const path = model.endsWith('.json') ? model : modelLoader(model).fileInDirectory('model.json');
    this.model = await tf.loadLayersModel(path);
    if (this.audioContext) {
      await this.processStream();
    } else {
      throw new Error('Could not access microphone - getUserMedia not available');
    }
    return this;
  }

  async processStream(): Promise<void> {
    await tf.nextFrame();

    const mic = this.audioContext.createMediaStreamSource(this.stream);
    const minBufferSize = (this.audioContext.sampleRate / 16000) * 1024;
    let bufferSize = 4;
    while (bufferSize < minBufferSize) bufferSize *= 2;

    // TODO: replace deprecated methods using AudioWorklet
    const scriptNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    scriptNode.onaudioprocess = this.processMicrophoneBuffer.bind(this);
    const gain = this.audioContext.createGain();
    gain.gain.setValueAtTime(0, this.audioContext.currentTime);

    mic.connect(scriptNode);
    scriptNode.connect(gain);
    gain.connect(this.audioContext.destination);

    if (this.audioContext.state !== 'running') {
      console.warn('User gesture needed to start AudioContext, please click');
    }
  }

  async processMicrophoneBuffer(event: AudioProcessingEvent): Promise<void> {
    await tf.nextFrame();
    
    PitchDetection.resample(event.inputBuffer, (resampled) => {
      tf.tidy(() => {
        this.running = true;

        const centMapping = tf.add(tf.linspace(0, 7180, 360), tf.tensor(1997.3794084376191));
        const frame = tf.tensor(resampled.slice(0, 1024));
        const zeromean = tf.sub(frame, tf.mean(frame));
        // TODO: what is this supposed to be? Appears to be dividing an array by a number.
        const framestd = tf.tensor(tf.norm(zeromean).dataSync() / Math.sqrt(1024));
        const normalized = tf.div(zeromean, framestd);
        const input = normalized.reshape([1, 1024]);
        const activation = (this.model!.predict([input]) as tf.Tensor).reshape([360]);
        const confidence = activation.max().dataSync()[0];
        const center = activation.argMax().dataSync()[0];
        // TODO: why convert this to a string?
        this.results = { confidence: confidence.toFixed(3) };

        const start = Math.max(0, center - 4);
        const end = Math.min(360, center + 5);
        const weights = activation.slice([start], [end - start]);
        const cents = centMapping.slice([start], [end - start]);

        const products = tf.mul(weights, cents);
        const productSum = products.dataSync<'float32'>().reduce((a, b) => a + b, 0);
        const weightSum = weights.dataSync<'float32'>().reduce((a, b) => a + b, 0);
        const predictedCent = productSum / weightSum;
        const predictedHz = 10 * (2 ** (predictedCent / 1200.0));

        this.frequency = (confidence > 0.5) ? predictedHz : null;
      });
    });
  }

  /**
   * Returns the pitch from the model attempting to predict the pitch.
   * @param {function} callback - Optional. A function to be called when the model has generated content. If no callback is provided, it will return a promise that will be resolved once the model has predicted the pitch.
   * @returns {number}
   */
  public async getPitch(callback: ML5Callback<number | null>): Promise<number | null> {
    return callCallback( ( async () => {
      await this.ready;
      await tf.nextFrame();
      return this.frequency;
    })(), callback);
  }

  static resample(audioBuffer: AudioBuffer, onComplete: (samples: Float32Array) => void) {
    const interpolate = (audioBuffer.sampleRate % 16000 !== 0);
    const multiplier = audioBuffer.sampleRate / 16000;
    const original = audioBuffer.getChannelData(0);
    const subsamples = new Float32Array(1024);
    for (let i = 0; i < 1024; i += 1) {
      if (!interpolate) {
        subsamples[i] = original[i * multiplier];
      } else {
        const left = Math.floor(i * multiplier);
        const right = left + 1;
        const p = (i * multiplier) - left;
        subsamples[i] = (((1 - p) * original[left]) + (p * original[right]));
      }
    }
    onComplete(subsamples);
  }
}

// TODO: "If no callback is provided, it will return a promise that will be resolved once the model has loaded."
const pitchDetection = (modelPath = './', context: AudioContext, stream: MediaStream, callback: ML5Callback<PitchDetection>) => new PitchDetection(modelPath, context, stream, callback);

export default pitchDetection;
