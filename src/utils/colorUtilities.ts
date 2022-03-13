import type p5 from "p5";

export type RGB = [number, number, number];

/**
 * Convert a p5 Color object into an array of R,G,B values.
 * Note: moved from BodyPix.
 */
export const p5Color2RGB = (p5ColorObj: p5.Color): RGB => {
    const regExp = /\(([^)]+)\)/;
    const match = regExp.exec(p5ColorObj.toString('rgb'));
    if (!match) throw new Error(`invalid rgb value ${p5ColorObj.toString('rgb')}`)
    const [r, g, b] = match[1].split(',').map(parseFloat);
    return [r, g, b];
}

/**
 * Check if a color is a p5 Color object.
 * Look at the properties instead of using `instanceof` in case p5 is not loaded.
 */
export const isP5Color = (color: number[] | p5.Color): color is p5.Color => {
    return 'setBlue' in color;
}

/**
 * Check if a color is a tuple with 3 (or more) numbers.
 */
export const isRgbColor = (color: number[] | p5.Color): color is RGB => {
    return Array.isArray(color) && color.length >= 3;
}

/**
 * Convert an array or a p5 Color object into an RGB tuple,
 * throwing an error if the input is malformed.
 */
export const toRgb = (color: number[] | p5.Color): RGB => {
    if (isRgbColor(color)) return color;
    if (isP5Color(color)) return p5Color2RGB(color);
    throw new Error(`Invalid color ${color}. Color must be an array [number, number, number] or a p5.Color object.`);
}
