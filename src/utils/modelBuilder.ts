import {ArgumentValidator, classifyArguments, ValidatedResults} from "./argumentValidator";

export type AsArgs<ArgMap extends Record<string, ArgumentValidator<any>>> = {
    [K in keyof ArgMap]: ReturnType<ArgMap[K]['validate']>;
}

interface CreateModelSettings<Model, Instance, Options extends {}, ArgMap extends Record<string, ArgumentValidator<any>> & {options: Options}> {
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
        return buildModel({...assigned, options: mergedOptions});
    }
}