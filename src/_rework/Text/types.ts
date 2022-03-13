import { ModelMeta } from "../Image/types";

export interface TextModelTasks<T, O> {
  classify?: (text: string, options?: O) => Promise<T>;
  generate?: (text: string, options?: O) => Promise<T>;
}

export type Task = keyof TextModelTasks<any, any>;

export interface TextModel<T, O> extends TextModelTasks<T, O>, ModelMeta<O> {
}

export type TextModelConstructor<T, O> = (options?: Partial<O>) => TextModel<T, O>;

export type ReturnFrom<M> = M extends TextModelTasks<infer T, any> ? T : never;

export type OptionsFrom<M> = M extends TextModelTasks<any, infer O> ? O : never;

export const textModel = <M extends TextModel<any, any>>(
  builder: (options?: Partial<OptionsFrom<M>>) => M
) => builder;
