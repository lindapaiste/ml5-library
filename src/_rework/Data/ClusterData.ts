import * as tf from "@tensorflow/tfjs";
import { DataPoint } from "./DataLoader";

export interface InternalDataPoint {
  tensor: tf.Tensor1D;
  isCorePoint?: boolean;
  cluster?: number;
}

export default class ClusterData {

  /**
   * Store the data in its original format.
   */
  public readonly original: DataPoint[];

  /**
   * An array of data point objects to be updated by the clustering algorithm.
   * Contains a 1D Tensor with the values for this data point
   * and the index of the cluster that it has been assigned to.
   */
  public dataset: InternalDataPoint[];

  /**
   * Combined 2D Tensor of all data points.
   */
  public dataTensor: tf.Tensor2D;

  /**
   * Pass the raw data to the constructor and create Tensors.
   */
  constructor(original: DataPoint[]) {
    this.original = original;
    this.dataTensor = tf.tensor2d(original.map(d => Object.values(d)));
    this.dataset = original.map(d => ({
      tensor: tf.tensor1d(Object.values(d))
    }));
  }

  /**
   * Filter the data points to just the ones matching a particular cluster.
   */
  public getClusterPoints(clusterIndex: number | undefined): InternalDataPoint[] {
    return this.dataset.filter(d => d.cluster === clusterIndex);
  }

  /**
   * Compute the centroid of a cluster based on its data points.
   * Will return undefined if the cluster is empty.
   */
  public findClusterCentroid(clusterIndex: number | undefined): tf.Tensor1D | undefined {
    const points = this.getClusterPoints(clusterIndex);
    if (points.length === 0) return undefined;
    return tf.tidy(() => {
      return tf.stack(points.map(d => d.tensor)).mean<tf.Tensor1D>(0);
    })
  }

  /**
   * Dispose of all Tensors.
   */
  public dispose(): void {
    this.dataTensor.dispose();
    this.dataset.forEach(d => d.tensor.dispose());
  }
}
