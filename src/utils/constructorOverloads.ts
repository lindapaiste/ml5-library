import type { MediaElement } from 'p5';
import { ML5Callback } from "./callcallback";
import handleArguments from "./handleArguments";

interface VideoOptionsInstance {
  ready: Promise<this>;
}

interface VideoOptionsConstructor<InstanceType extends VideoOptionsInstance, OptionsType extends object> {
  new(video?: HTMLVideoElement | MediaElement, options?: Partial<OptionsType>, callback?: ML5Callback<InstanceType>): InstanceType;
}

function videoOptionsCallback<InstanceType extends VideoOptionsInstance, OptionsType extends object>(
  ClassConstructor: VideoOptionsConstructor<InstanceType, OptionsType>
) {
  function create(callback: ML5Callback<InstanceType>): InstanceType;
  function create(video: HTMLVideoElement | MediaElement, callback: ML5Callback<InstanceType>): InstanceType;
  function create(options: OptionsType, callback: ML5Callback<InstanceType>): InstanceType;
  function create(video: HTMLVideoElement | MediaElement, options: OptionsType, callback: ML5Callback<InstanceType>): InstanceType;
  function create(): Promise<InstanceType>;
  function create(video: HTMLVideoElement | MediaElement): Promise<InstanceType>;
  function create(options: OptionsType): Promise<InstanceType>;
  function create(video: HTMLVideoElement | MediaElement, options: OptionsType): Promise<InstanceType>;
  function create(...args: (HTMLVideoElement | MediaElement | OptionsType | ML5Callback<InstanceType>)[]) {
    const { video, options = {}, callback } = handleArguments(...args);
    // Note: error because options type can technically be a function
    const instance = new ClassConstructor(video, options, callback as any);
    return callback ? instance : instance.ready;
  }
  return create;
}

export default {
  videoOptionsCallback
}
