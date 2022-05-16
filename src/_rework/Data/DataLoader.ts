import * as tf from "@tensorflow/tfjs";
import type { CSVConfig } from '@tensorflow/tfjs-data/dist/types';
import * as tfvis from '@tensorflow/tfjs-vis'

export type DataPoint<T = number> = T[] | Record<PropertyKey, T>

export default class DataLoader<T = number> {
  private featureKeys: string[] = [];
  private labelKeys: string[] = [];

  constructor() {
  }

  // TODO: handle JSON file and Blob
  async load(inputData: string | DataPoint<T>[]): Promise<T[][]> {
    let data: DataPoint<T>[];
    if (typeof inputData === 'string') {
      // TODO: handle non-numeric CSV inputs -- cast to a numeric type like float32
      data = await this.loadCsv(inputData);
    } else {
      data = inputData;
    }
    return data.map((d) => Object.values(d));
  }

  private async loadCsv(path: string, config?: CSVConfig) {
    const myCsv = tf.data.csv(path, config);
    // TODO: use iterator instead of converting to an array.
    return myCsv.toArray();
  }
}

async function fromtfVisDocs () {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
      tf.layers.dense({ units: 10, activation: 'softmax' }),
    ]
  });

  model.compile({
    optimizer: 'sgd',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  const data = tf.randomNormal([100, 784]);
  const labels = tf.randomUniform([100, 10]);

  function onBatchEnd(batch, logs) {
    console.log('Accuracy', logs.acc);
  }

  const surface = { name: 'show.history', tab: 'Training' };
  // Train for 5 epochs with batch size of 32.
  const history = await model.fit(data, labels, {
    epochs: 5,
    batchSize: 32
  });

  tfvis.show.history(surface, history, ['loss', 'acc']);
}
