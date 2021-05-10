import {Color} from "p5";

export type P5Color = Color;
export type RGB = [number, number, number];
export type RGBA = [number, number, number, number];

export const isP5Color = (color: object): color is P5Color => {
    return "getAlpha" in color;
}

export const p5ColorToRGBA = (color: P5Color): RGBA => {
    // There is already an array in the property Color.levels, but this is not intended to be "public"
    // @ts-ignore
    return color.levels;
    // If not wanting to access the private property, can do:
    // return [red(color), green(color), blue(color)]
}

export const p5ColorToRGB = (color: P5Color): RGB => {
    // @ts-ignore
    return color.levels.slice(0,3);
}