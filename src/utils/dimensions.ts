export type HasWidth = {
    width: number;
} | {
    getWidth(): number;
}

export type HasHeight = {
    height: number;
} | {
    getHeight(): number;
}

export type HasDimensions = HasWidth & HasHeight;

export const getHeight = (object: HasHeight): number => {
    if ( "getHeight" in object ) {
        return object.getHeight();
    } else return object.height;
}

export const getWidth = (object: HasWidth): number => {
    if ( "getWidth" in object ) {
        return object.getWidth();
    } else return object.width;
}

/**
 * @typedef {Object} Dimensions
 * @property {number} width
 * @property {number} height
 */
export interface Dimensions {
    width: number;
    height: number;
}

export const getDimensions = (object: HasDimensions): Dimensions => ({
    width: getWidth(object),
    height: getHeight(object),
});
