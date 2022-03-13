// Callback handled externally in model runner callback: (err: any, res: T) => void

import { InputImage } from "../../utils/handleArguments";

export interface ImageDetector<T, O> {
  detect: (img: InputImage, options: O) => Promise<T>;
}

export interface ImageGenerator<T, O> {
  generate: (img: InputImage, options: O) => Promise<T>;
}

export interface ImageStylizer<T, O> {
  stylize: (img: InputImage, options: O) => Promise<T>;
}

export interface ImageSegmenter<T, O> {
  segment: (img: InputImage, options: O) => Promise<T>;
}

export type ImageModelUnion<T, O> =
  ImageDetector<T, O>
  | ImageGenerator<T, O>
  | ImageStylizer<T, O>
  | ImageSegmenter<T, O>

export interface ImageModelTasks<T, O> {
  classify?: (img: InputImage, options: O) => Promise<T>;
  detect?: (img: InputImage, options: O) => Promise<T>;
  generate?: (img: InputImage, options: O) => Promise<T>;
  stylize?: (img: InputImage, options: O) => Promise<T>;
  segment?: (img: InputImage, options: O) => Promise<T>;
}

export type Task = keyof ImageModelTasks<any, any>;

export interface ModelMeta<O> {
  name: string;
  event: string;
  trainable?: boolean;
  defaultOptions: O;
}

export interface ImageModel<T, O> extends ImageModelTasks<T, O>, ModelMeta<O> {
}

export type ImageModelConstructor<T, O> = (options: O) => ImageModel<T, O>;

export type ReturnFrom<M> = M extends ImageModelTasks<infer T, any> ? T : never;

export type OptionsFrom<M> = M extends ImageModelTasks<any, infer O> ? O : never;

export type Unpromise<T> = T extends Promise<infer U> ? U : T;
