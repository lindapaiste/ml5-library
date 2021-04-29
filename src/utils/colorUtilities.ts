import {Color} from "p5";

type RGB = [number, number, number];
type RGBA = [number, number, number, number];


export const runP5Function = (f: (p5: p5))

export const isP5Color = (color: object): color is Color => {
    return "getAlpha" in color;
}

export const p5ColorToRGB = (color: Color): RGB => {
    // is already an array in the property Color.levels, but this is not intended to be "public"
    return color.levels;
}