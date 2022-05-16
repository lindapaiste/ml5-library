/**
 * Helper for interacting with the results of a KMeans or DBScan clustering.
 * Note: Can potentially be used at each iteration.
 */
import * as tf from '@tensorflow/tfjs';
import ClusterData from "./ClusterData";
import { DataPoint } from "./DataLoader";

class __Cluster {
  private _centroid: DataPoint | undefined;

  public index: number;

  public label: string;

  public dataPoints: Datum[];

  constructor(args: {
    index: number;
    label?: string;
    dataPoints: Datum[];
    centroid?: DataPoint;
  }) {
    this._centroid = args.centroid;
    this.index = args.index;
    this.label = args.label ?? (args.index + 1).toString(10);
    this.dataPoints = args.dataPoints;
  }

  /**
   * Lazily access the centroid.
   * If passed into the constructor, return that value.
   * Otherwise compute it the first time it is accessed and store for later.
   */
  get centroid(): DataPoint {
    if (this._centroid) {
      return this._centroid;
    }
    if (this.dataPoints.length === 0) {
      throw new Error('Cannot compute the centroid of a cluster with 0 data points.');
    }
    const values = tf.tidy(() => {

    })
    // TODO: apply labels
  }
}

interface Cluster {
  centroid?: number[];
  index: number | undefined;
  label?: string;
}

export interface ClusterOptions {
  returnTensors?: boolean;
  returnCentroids?: boolean;
}

interface Datum<T = DataPoint> {
  data: T;
  index: number;
}

export interface ClusterResultArgs<M extends { clusterCount: number; } = { clusterCount: number; [metric: string]: number }> {
  data: ClusterData;
  config: ClusterOptions;
  metrics: { clusterCount: number; };
  centroids?: number[][];
}

export interface ClusterCreator extends ClusterResultArgs {
  fit(): void;

  dispose(): void;
}

export default class ClusterResult {
  /**
   * The values which were provided to the clustering algorithm.
   */
  public originalData: DataPoint[];
  /**
   * Include the 2D Tensor of data if `returnTensors` is true.
   */
  public dataTensor?: tf.Tensor2D;
  /**
   * Array of cluster indices for each data point.
   */
  public predictedIndices: (number | undefined)[];
  /**
   * Total amount of computed clusters.
   * Will equal the provided `k` for KMeans, but varies for DBSCAN.
   */
  public clusterCount: number;
  /**
   * Array of centroid points for each cluster.
   * Will be undefined unless `returnCentroids` is true.
   */
  public centroids?: number[][];
  /**
   * Metrics associated with the accuracy of the clustering.
   */
  public metrics: Record<string, number>;

  constructor({ data, config, metrics, centroids }: ClusterResultArgs) {
    this.clusterCount = metrics.clusterCount;
    this.originalData = data.original;
    this.predictedIndices = data.dataset.map(d => d.cluster);
    this.metrics = metrics;
    if (config.returnTensors) {
      this.dataTensor = data.dataTensor;
    }
    if (config.returnCentroids) {
      if (centroids) {
        this.centroids = centroids;
      } else {
        this.centroids = [];
        for (let i = 0; i < this.clusterCount; i++) {
          const tensor = data.findClusterCentroid(i);
          if (tensor) {
            // Cast from TypedArray to number[]
            this.centroids[i] = Array.from(tensor.dataSync());
            tensor.dispose();
          }
        }
      }
    }
  }

  /**
   * Returns the data points in the order that they were provided,
   * along with the index and center of the assigned cluster.
   *
   * The cluster can be undefined for DBSCAN if this is a "noise" point.
   * For KMeans, the cluster will always be defined.
   */
  get predictions(): Array<Datum & { cluster: number | undefined }> {
    return this.originalData.map((data, dataIndex) => {
      const clusterIndex = this.predictedIndices[dataIndex];
      // const center = this.centers[clusterIndex];
      return {
        data,
        index: dataIndex,
        cluster: clusterIndex
      }
    });
  }

  /**
   * Get the results grouped by cluster.
   */
  get clusters(): Array<Cluster & { dataPoints: Datum[] }> {
    const clusters: Array<Cluster & { dataPoints: Datum[] }> = [];
    for (let i = 0; i < this.clusterCount; i++) {
      clusters.push({
        index: i,
        centroid: this.centroids?.[i],
        dataPoints: []
      })
    }
    this.predictedIndices.forEach((clusterIndex, dataIndex) => {
      if (clusterIndex) { // Skip over noise.
        clusters[clusterIndex].dataPoints.push({
          data: this.originalData[dataIndex],
          index: dataIndex
        })
      }
    })
    return clusters;
  }

  /**
   * Get all data points which were not assigned to a cluster.
   * For DBSCAN only, as KMeans assigns all points.
   */
  get noisePoints(): Datum[] {
    const noise: Datum[] = [];
    this.predictedIndices.forEach((clusterIndex, dataIndex) => {
      if (clusterIndex === undefined) {
        noise.push({
          data: this.originalData[dataIndex],
          index: dataIndex
        })
      }
    });
    return noise;
  }
}
