import {keysValuesToObject} from "./objectUtilities";

/**
 * Convert a CSV data string into an array of objects for each row
 * where the keys are the headers and the values are a string or number from the cell.
 * @param csv
 */
export const csvToJSON = (csv: string): Record<string, string | number>[] => {
    // Split the string by linebreak
    const [header, ...lines] = csv.split('\n');
    // Split the cells by commas
    const labels = header.split(',');
    // Iterate through every row
    return lines.map( line => {
        // Split the current line into an array
        const rawValues = line.split(',');
        // See if each value can be converted to a number, otherwise keep the string.
        const values = rawValues.map(value => {
            const valueAsNum = Number(value);
            return isNaN(valueAsNum) ? value : valueAsNum;
        });
        // For each header, create a key/value pair
        return keysValuesToObject(labels, values);
    });
}