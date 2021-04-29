/* eslint max-len: "off" */

import * as tf from '@tensorflow/tfjs';

// TODO: modernize code

export default class CheckpointLoaderPix2pix {
  private readonly urlPath: string;
  private weightsCache: Record<string | number, tf.Tensor> | null;

  constructor(urlPath: string) {
    this.urlPath = urlPath;
    this.weightsCache = null;
  }

  getAllVariables(): Promise<Record<string | number, tf.Tensor>> {
    return new Promise((resolve, reject) => {
      if (this.weightsCache !== null) {
        resolve(this.weightsCache);
        return;
      }

      const xhr = new XMLHttpRequest();
      xhr.open('GET', this.urlPath, true);
      xhr.responseType = 'arraybuffer';
      xhr.onload = () => {
        if (xhr.status !== 200) {
          reject(new Error('missing model'));
          return;
        }
        const buf: ArrayBuffer = xhr.response;
        if (!buf) {
          reject(new Error('invalid arraybuffer'));
          return;
        }

        const parts: ArrayBuffer[] = [];
        let offset = 0;
        while (offset < buf.byteLength) {
          const b = new Uint8Array(buf.slice(offset, offset + 4));
          offset += 4;
          const len = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]; // eslint-disable-line no-bitwise
          parts.push(buf.slice(offset, offset + len));
          offset += len;
        }

        const shapes: ({name: string; shape: number[];})[] = JSON.parse((new TextDecoder('utf8')).decode(parts[0]));
        const index = new Float32Array(parts[1]);
        const encoded = new Uint8Array(parts[2]);

        // decode using index
        const arr = new Float32Array(encoded.length);
        for (let i = 0; i < arr.length; i += 1) {
          arr[i] = index[encoded[i]];
        }

        const weights: Record<string | number, tf.Tensor> = {};
        offset = 0;
        for (let i = 0; i < shapes.length; i += 1) {
          const { shape } = shapes[i];
          const size = shape.reduce((total, num) => total * num);
          const values = arr.slice(offset, offset + size);
          const tfarr = tf.tensor1d(values, 'float32');
          weights[shapes[i].name] = tfarr.reshape(shape);
          offset += size;
        }
        this.weightsCache = weights;
        resolve(weights);
      };
      xhr.send(null);
    });
  }
}
