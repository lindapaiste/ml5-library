import {extractImageElement, extractVideoElement, TfImageSource} from "./imageUtilities";
import {getModelPath} from "./modelLoader";

export interface ArgumentValidator<T> {
    /**
     * The property key to assign to matching values in the returned object, such as 'video' or 'callback'
     */
    property: string;
    /**
     * Text shown in error messages, such as 'an HTML video element or p5 video element'
     */
    textDescription: string;
    /**
     * Function which checks whether an unknown argument is of this type.  Can perform both checking (`instanceof`)
     * and normalization (returning the .elt property on p5 elements).  Return a valid value is checks are passed, or
     * `undefined` if not a match.  Responses are expected to be mutually exclusive.  If ambiguous, must use
     * `excludeOthers` flag to prevent multiple matches.
     */
    validate: (arg: any) => T | undefined;
    /**
     * If `true`, only assign an argument to this type if the `validator` check returns `undefined` for all other potential
     * properties.  This is a way to avoid multiple matches with the 'options' argument which is any object.
     */
    excludeOthers?: boolean;
}

const video: ArgumentValidator<HTMLVideoElement> = {
    property: 'video',
    textDescription: 'an HTML video element or p5 video element',
    validate: extractVideoElement,
}
const callback: ArgumentValidator<Function> = {
    property: 'callback',
    textDescription: 'a callback function',
    validate: (arg: any) => {
        if (typeof arg === "function") {
            return arg;
        }
    }
}

const image: ArgumentValidator<TfImageSource> = {
    property: 'image',
    textDescription: 'an HTMLImageElement, HTMLCanvasElement, HTMLVideoElement, ImageData object, or p5 image element',
    validate: extractImageElement,
}

const options: ArgumentValidator<object> = {
    property: 'options',
    textDescription: 'an options object',
    validate: (arg: any) => {
        // truthy check removes null
        if (arg && typeof arg === "object") {
            return arg;
        }
    },
    excludeOthers: true,
}

const modelPath = (extension: string): ArgumentValidator<string> => ({
    property: 'modelPath',
    textDescription: `a url of a model in ${extension} format`,
    validate: (arg: any) => {
        if (typeof arg === "string" && arg.endsWith(extension)) {
            return getModelPath(arg);
        }
    },
})

const modelName = (acceptedNames: string[]): ArgumentValidator<string> => ({
    property: 'modelName',
    textDescription: 'the name of a predefined model',
    validate: (arg: any) => {
        // should I check includes here?  what about warnings?
        if (typeof arg === "string" && acceptedNames.includes(arg)) {
            return arg;
        }
    }
});

const number: ArgumentValidator<number> = {
    property: 'number',
    textDescription: 'a number', // TODO: a number of what? varies by model
    validate: (arg: any) => {
        if (typeof arg === "number") {
            return arg;
        }
    }
}

export const ARGS = {
    video,
    image,
    callback,
    options,
    modelName,
    modelPath,
    number
}

// extends Record<string, ArgumentValidator<any>>
export type ValidatedResults<ValidatorsMap> = {
    [K in keyof ValidatorsMap]?: ValidatorsMap[K] extends ArgumentValidator<infer V> ? V : never;
}

export class InvalidArgumentError extends TypeError {
    public readonly arg: any;

    constructor(arg: any, validators: ArgumentValidator<any>[], i?: number) {
        const message = `Invalid argument${i === undefined ? '.' : `in position ${i} (zero-indexed).`}.
      Received value: ${String(arg)}.
      Argument must be one of the following types:\n
      ${validators.map(o => `'${o.property}' - ${o.textDescription}`).join('\n')}
      `;
        super(message);
        this.name = 'InvalidArgumentError';
        this.arg = arg;
    }
}

const createArgumentClassifier = <ValidatorsMap extends Record<string, ArgumentValidator<any>>>(validators: ValidatorsMap) => {
    return class {
        constructor(...args: any[]) {
            args.forEach(this.addArg);
        }

        addArg(arg: any, index?: number): void {
            // store matches for which excludeOthers was true
            const potential: ArgumentValidator<any>[] = [];
            Object.entries(validators).forEach(([property, validator]) => {
                const value = validator.validate(arg);
                if (value !== undefined) {
                    if (validator.excludeOthers) {
                        potential.push(validator);
                    } else {
                        this[property] = value;
                        return;
                    }
                }
            });
            if (potential.length === 1) {
                // TODO assign property
                return;
            } else if (potential.length > 1) {
                throw new Error("Ambiguous argument"); //TODO
            } else {
                // Notify user about invalid arguments (would be ok to just skip)
                // But skip over falsey arguments assuming that these are omissions
                if (arg) {
                    throw new InvalidArgumentError(arg, Object.values(validators), index);
                }
            }
        }
    }
}

export const classifyArguments = <ValidatorsMap extends Record<string, ArgumentValidator<any>>>(validators: ValidatorsMap, ...args: any[]): ValidatedResults<ValidatorsMap> => {
    const validated: ValidatedResults<ValidatorsMap> = {};

    args.forEach((arg, index) => {
        // store matches for which excludeOthers was true
        const potential: ArgumentValidator<any>[] = [];
        Object.entries(validators).forEach(([property, validator]) => {
            const value = validator.validate(arg);
            if (value !== undefined) {
                if (validator.excludeOthers) {
                    potential.push(validator);
                } else {
                    validated[property] = value;
                    return;
                }
            }
        });
        if (potential.length === 1) {
            // TODO assign property
            return;
        } else if (potential.length > 1) {
            throw new Error("Ambiguous argument"); //TODO
        } else {
            // Notify user about invalid arguments (would be ok to just skip)
            // But skip over falsey arguments assuming that these are omissions
            if (arg) {
                throw new InvalidArgumentError(arg, Object.values(validators), index);
            }
        }
    });

    return validated;
}