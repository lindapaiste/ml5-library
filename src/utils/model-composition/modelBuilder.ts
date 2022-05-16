import {ArgumentValidator, classifyArguments, ValidatedResults} from "../argumentValidator";
import callCallback, {ML5Callback} from "../callcallback";

export type AsArgs<ArgMap extends Record<string, ArgumentValidator<any>>> = {
    [K in keyof ArgMap]: ReturnType<ArgMap[K]['validate']>;
}

interface CreateModelSettings<Model, Instance, Options, ArgMap> {
    acceptsArgs: ArgMap, // TODO: should add callback automatically,
    defaults: Options,
    buildModel: (args: ValidatedResults<ArgMap> & {options: Options}) => Promise<Model>,
    methods: {}
}

export const createModel = <Model, Instance, Options extends Record<string, any>, ArgMap extends Record<string, ArgumentValidator<any>> & {options: Options}>({
    acceptsArgs,
    defaults,
    buildModel,
    methods
}: CreateModelSettings<Model, Instance, Options, ArgMap>) => {
    const build = (...args: any[]): Promise<Model> => {
        const assigned = classifyArguments(acceptsArgs, ...args);
        const passedOptions = assigned.options as Partial<Options>;
        // TODO: merge with warnings about misnamed, invalid, etc
        const mergedOptions: Options = {
            ...defaults,
            ...passedOptions
        };
        return callCallback(buildModel({...assigned, options: mergedOptions}), assigned.callback);
    }
}

type Unpromise<T> = T extends Promise<infer U> ? U : T;
type Element<T> = T extends Array<infer U> ? U : never;

export const wrapAsyncMethod = <M extends (...args: any[]) => Promise<any>>(method: M) => {
    // add a callback argument
    // accept the args optionally and in any order
    return (...args: Array<Element<Parameters<M>> | ML5Callback<Unpromise<ReturnType<M>>>>) => {

    }
}
