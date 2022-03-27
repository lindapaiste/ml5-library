import {RawData, RawDataRow, RawPropertyData, UnlabeledData, XsOrYs} from "./NeuralNetworkData";
import {Input} from "./index";
import {isNumberArray, isTypedArray, TypedArray} from "../utils/arrayUtilities";
import {SyncConvertible, toSyncConverter} from "../utils/imageConversion";
import {keysValuesToObject} from "../utils/objectUtilities";
import {LabelConfig, LabelManager, XOrYLabelManager, xyAsText} from "./labels";

/**
 * Manages separating mixed data rows into xs and ys based on property names or indexes.
 *
 * Stores data as arrays rather than keyed objects because:
 * 1. it is easier to remove labels from an object then to add labels to an array.
 * 2. labels can be set after some rows are already added.
 */
export class XandYRowDataAdder {

    private _raw: UnlabeledData = [];

    constructor(public labels: LabelManager) {
    }

    get raw(): UnlabeledData {
        return this._raw;
    }

    /**
     * Check a variable to see if it looks like a data row.
     * Can be: an array (check values?), a keyed object,
     * @param row
     */
    public isValidData(row: any) {

    }

    /**
     * Add an object which contains both input and output properties.
     * Pick the xs and ys from their labels.
     * From a JSON file, etc.
     */
    private addObjectRow(row: Record<string, RawPropertyData>) {
        if (!this.labels.xs.names || !this.labels.ys.names) {
            throw new Error("Cannot add an object without specifying which properties to use as inputs and outputs. Provide an array of labels for 'inputs' and 'outputs' in your neural network options.");
        }
        const xs = this.labels.xs.extractProperties(row);
        const ys = this.labels.ys.extractProperties(row);
        this._raw.push({xs, ys});
    }

    /**
     * Add an array of unlabeled data. Slice based on the count of xs/ys.
     * Expects that the order of inputs matches the order of the provided input/output labels.
     */
    private addArrayRow(row: RawPropertyData[]) {
        if (!this.labels.xs.count || !this.labels.ys.count) {
            throw new Error("Missing configuration options for 'inputs' and 'outputs'. In order to add an array of values you must provide either an array of labels or a number to the 'inputs' and 'outputs' options when creating your neural network.");
        }
        const expectedLength = this.labels.xs.count + this.labels.ys.count;
        if (row.length < expectedLength) {
            throw new Error(`Insufficient values provided. Expected length ${expectedLength}: ${this.labels.xs.count} inputs and ${this.labels.ys.count} outputs. Received an array with length ${row.length}.`);
        }
        // Too many values is just a warning, but could be an error.
        if (row.length > expectedLength) {
            console.warn(`Received more values than the expected count of ${expectedLength}: ${this.labels.xs.count} inputs and ${this.labels.ys.count} outputs. Will use the first ${expectedLength} values and ignore the remaining ${row.length - expectedLength}.`);
        }
        const xs = row.slice(0, this.labels.xs.count);
        const ys = row.slice(this.labels.xs.count, expectedLength);
        this._raw.push({xs, ys});
    }

    /**
     * Add an object which already has the xs and ys separated and labeled.
     */
    private addXYObject(row: RawDataRow) {
        this.addUserData(row.xs, row.ys);
    }

    /**
     * Public method can handle a variety of formats.
     */
    public addRow(row: RawDataRow | RawPropertyData[] | Record<string, RawPropertyData>) {
        if (Array.isArray(row)) {
            this.addArrayRow(row);
        } else if (row.xs && row.ys) {
            this.addXYObject(row);
        } else {
            this.addObjectRow(row);
        }
    }

    /**
     * Handles looping and also adds an index to error messages.
     */
    public addManyRows(rows: Array<RawDataRow | RawPropertyData[] | Record<string, RawPropertyData>>) {
        rows.forEach((row, idx) => {
            try {
                this.addRow(row);
            } catch (error) {
                // Catch error in order to re-throw with line number
                throw new Error(`Error in line #${idx + 1}:\n ${error?.message}`);
            }
        });
    }

    /**
     * Data added by the user through calling `addData` on the neural network will already have the xs and ys separated.
     */
    public addUserData(xs: RawPropertyData[] | Record<string, RawPropertyData>, ys: RawPropertyData[] | Record<string, RawPropertyData>) {
        this._raw.push({
            xs: this.processUserDataXsOrYs(xs, 'xs'),
            ys: this.processUserDataXsOrYs(ys, 'ys'),
        })
    }


    /**
     * Create labels from indexes if no texts have been provided
     */
    private createLabels(xsOrYs: XsOrYs): string[] {
        const {count, names} = this.labels[xsOrYs];
        if (!count) {
            throw new Error("Neither labels nor counts have been provided.");
        }
        return names || Array.from({length: count}, (_, i) => `${xyAsText(xsOrYs)}_${i}`);
    }

    /**
     * Add labels if possible when exporting to JSON.
     */
    public prepareForSave(): RawData {
        const xLabels = this.createLabels('xs');
        const yLabels = this.createLabels('ys');
        return this.raw.map(({xs, ys}) => ({
            xs: keysValuesToObject(xLabels, xs),
            ys: keysValuesToObject(yLabels, ys),
        }))
    }
}

export class StandardInputPreparer {
    constructor(private labels: XOrYLabelManager) {
    }

    /**
     * Convert to an array. Order object properties based on the order of labels. Handle errors.
     * Can set the label config based on data if none was provided.
     */
    public prepareInput(data: RawPropertyData[] | Record<string, RawPropertyData>): RawPropertyData[] {
        const count = this.labels.count;
        if (Array.isArray(data)) {
            // If no count has been set before, can set the count now to ensure that future rows match.
            if (!count) {
                this.labels.setCount(data.length);
                console.log(`Set expected count for ${this.labels.inOutPlural} to ${data.length} based on the length of the provided data array.`);
                return data;
            }
            // Error if not enough values.
            else if (data.length < count) {
                throw new Error(`Insufficient ${this.labels.inOutSingular} values provided. Expected length ${count}. Received an array with length ${data.length}.`);
            }
            // Too many values is just a warning, but could be an error.
            else if (data.length > count) {
                console.warn(`Received more ${this.labels.inOutSingular} values than the expected count of ${count}. Will use the first ${count} values and ignore the remaining ${data.length - count}.`);
                return data.slice(0, count);
            }
            // Correct count
            else return data;
        } else {
            // If labels are known, pick the values such that they match the order of the labels.
            if (this.labels.names) {
                return this.labels.extractProperties(data);
            }
            // Set the label names based on the property names.
            // If a count is known but not labels, then it must match the count of properties.
            const keys = Object.keys(data);
            if (count && keys.length !== count) {
                throw new Error(`Invalid data. Provided count in options.${this.labels.inOutPlural} does not match the count of properties on the current object. When provided data as keyed objects, you should set options.${this.labels.inOutPlural} to an array of label names.`);
            }
            this.labels.setNames(keys);
            return this.labels.extractProperties(data);
        }
    }

}


type ImageInput = SyncConvertible | Uint8ClampedArray;

/**
 * Images don't have property names. Instead they have an input shape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS].
 * Sometimes they might be added through addData with a property name, like addData({image: element}, {label: 'something'});
 *
 * If an Image element is added, then the dimensions can be inferred.  If only pixel arrays are added then the
 * input shape must be known in advance, through the NN options.
 */
export class ImageInputPreparer {

    constructor(private labels: XOrYLabelManager) {
    }

    get shape(): number[] | undefined {
        return this.labels.shape;
    }

    get propertyName(): string | undefined {
        return this.labels.names?.[0];
    }

    get expectedLength(): number | undefined {
        if (this.shape) {
            const [width, height, channels] = this.shape;
            return width * height * channels;
        }
    }

    private imageToArray(image: ImageInput): number[] | TypedArray {
        if (isTypedArray(image) || Array.isArray(image) && isNumberArray(image)) {
            // If a pixel array is provided.
            if (!this.expectedLength) {
                throw new Error("No input shape provided. In order to pass an array of pixels, you must specify the 'inputs' option of your neural network as an array of three numbers in the form: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]");
            }
            if (this.expectedLength !== image.length) {
                throw new Error(`Invalid pixel array length. All images must be the same size and must match the dimensions specified in in 'options.inputs' property of the neural network. Expected pixel count ${this.expectedLength} but got pixel count ${image.length}`);
            }
            return image;
        } else {
            // Handle an image element.
            const converter = toSyncConverter(image);
            const pixels = converter.toPixels();
            // If dimensions are known, they must match.
            if ( this.shape ) {
                const [width, height, channels] = this.shape;
                // TODO: can the image be resized if the aspect ratio is correct?
                if (converter.getWidth() !== width || converter.getHeight() !== height) {
                    throw new Error(`Invalid image size. All images must have the same dimensions. Expected width ${width} and height ${height} but got width ${converter.getWidth()} and height ${converter.getHeight()}`);
                }
                if (channels < 1 || channels > 4) {
                    throw new Error(`Invalid channel count. Channels must be a number between 1 and 4. It is typically 3 (RGB) for an full-color image or 4 (RGBA) for an image with transparency.`);
                }
                // Slice the pixels array to the correct length based on channels.
                return pixels.slice(0, width * height * channels);
                // TODO: proper handling of 1 channel for grayscale
            }
            // Can set the dimensions if not already known.
            else {
                const width = converter.getWidth();
                const height = converter.getHeight();
                const channels = pixels.length / (width * height); // is probably 4 - RGBA
                this.labels.setShape([width, height, channels]);
                return pixels;
            }
        }
    }

    private extractImage(input: ImageInput | [ImageInput] | Record<string, ImageInput>): ImageInput {
        if (this.propertyName && this.propertyName in input) {
            input[this.propertyName];
        }
        if (Array.isArray(input) && input.length === 1) {
            return input[0];
        }
        return input;
    }

    /**
     * Expect either an image or an object with a labeled property which is an image.
     * Possibly an array with one element which is an image?
     * Not sure if image data is ever passed in via CSV.
     */
    public prepareInput(input: ImageInput | [ImageInput] | Record<string, ImageInput>): number[] {
        return [...this.imageToArray(this.extractImage(input))];
    }
}

interface InputPreparer {
    prepareInput(input: Input): RawPropertyData;
}