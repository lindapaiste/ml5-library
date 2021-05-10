import {RawPropertyData, XsOrYs} from "./NeuralNetworkData";
import {NNTask} from "./tasks";
import {NeuralNetworkOptions} from "./index";
import {isNumberArray} from "../utils/arrayUtilities";

export interface LabelConfig {
    names?: string[];
    count?: number;
    shape?: number[]; //tf.Shape;
}

/**
 * convert strings "xs"/"ys" to "inputs"/"outputs" for use in error messages
 * @param xsOrYs
 */
export const xyAsText = (xsOrYs: XsOrYs) => {
    return xsOrYs === "xs" ? "inputs" : "outputs";
}

/**
 * Labels are initially set based on the user-provided options,
 * but missing values can be filled in while adding data.
 */
export class LabelManager {
    public readonly xs: XOrYLabelManager;
    public readonly ys: XOrYLabelManager;

    constructor(private readonly task: NNTask, options: Partial<NeuralNetworkOptions> = {}) {
        const {inputs, outputs} = options;
        this.xs = new XOrYLabelManager(inputs ? task.parseInputs(inputs) : {}, 'xs');
        this.ys = new XOrYLabelManager(outputs ? task.parseInputs(outputs) : {}, 'ys');
    }

    /**
     * Gets the raw data only from both xs and ys.
     */
    get raw() {
        return {
            xs: this.xs.raw,
            ys: this.ys.raw
        }
    }
}

export class XOrYLabelManager {

    public readonly inOutSingular: "input" | "output";
    public readonly inOutPlural: "inputs" | "outputs";

    constructor(private _config: LabelConfig = {}, public readonly xsOrYs: XsOrYs) {
        this.inOutSingular = xsOrYs === "xs" ? "input" : "output";
        this.inOutPlural = xsOrYs === "xs" ? "inputs" : "outputs";
    }

    get raw(): LabelConfig {
        return this._config;
    }

    get names(): string[] | undefined {
        return this._config.names;
    }

    get count(): number | undefined {
        return this._config.count;
    }

    get shape(): number[] | undefined {
        return this._config.shape;
    }

    /**
     * Make sure that properties are pulled in the correct order.
     * Helper throws an error when a property is not defined on the object.
     */
    extractProperties<T>(obj: Record<string, T>): T[] {
        if ( ! this.names) {
            throw new Error(`No property names defined for ${this.inOutPlural}`);
        }
        return this.names.map(property => {
                if (!(property in obj)) {
                    throw new Error(`Property ${property} is missing in object ${JSON.stringify(obj)}.`);
                }
                return obj[property];
            });
    }

    /**
     * Throw an error if the array length does not match the specified count.
     * TODO: unsure how shape should come into play here.
     */
    validateLength(arr: any[]): boolean {
    }

    /**
     * Setting names also updates the count.
     */
    setNames(names: string[]): void {
        this._config.names = names;
        this._config.count = names.length;
    }

    setShape(shape: number[]): void {
        // TODO: should shape update the count? Does an image with a name have count 1 or w*h*c?
        this._config.shape = shape;
    }

    setCount(count: number): void {
        this._config.count = count;
    }
}

/**
 * Handle the many different formats for inputs and outputs options by assigning the value to named properties.
 *
 * Utility function defines the standard formatting to avoid repetition.
 * Can be overwritten for a particular task.
 */
export const basicParseInputsOutputs = (provided: number | string[] | number[]): LabelConfig => {
    if (Array.isArray(provided)) {
        if (provided.length === 0) {
            // Should an error be thrown on an empty array?
            return {};
        }
        if (isNumberArray(provided)) {
            // An array of numbers is assumed to be a shape rather than labels.  This is the format for images.
            // The count is the product of all shape dimensions. ie. [100, 100, 4] => 40,000
            // This is the expected length of an array provided to xs.
            return {
                count: provided.reduce((product, value) => value * product),
                shape: provided,
            };
        } else {
            // An array of strings are property names.
            return {
                count: provided.length,
                names: provided,
            }
        }
    } else if (typeof provided === "number") {
        // A number is a count of properties.
        // Note: this is overwritten in classification outputs, where a number is the number of options.
        return {
            count: provided,
        }
    } else {
        throw new Error("Invalid input or output configuration. Must provide an array of string labels or a single number.");
    }
}