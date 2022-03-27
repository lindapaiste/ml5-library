// Copyright (c) 2018 ml5
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

// Random

// Returns a random number between min (inclusive) and max (exclusive)
const randomFloat = (min = 0, max = 1) => (Math.random() * (max - min)) + min;

// Returns a random integer between min (inclusive) and max (inclusive)
const randomInt = (min = 0, max = 1) => Math.floor(Math.random() * ((max - min) + 1)) + min;

// Random Number following a normal distribution
// Taken from https://github.com/processing/p5.js/blob/master/src/math/random.js#L168
// Uses the Marsaglia polar method https://en.wikipedia.org/wiki/Marsaglia_polar_method
// Note 2021/05/03: updated this to include an inner and outer function as shown here
// https://stackoverflow.com/a/35599181/10431574 because the p5 method relies on `this`
// context of the prototype. Without that, `previous` and `y2` are always `undefined`.
const randomGaussian = (mean = 0, sd = 1) => {
    let y2: number;
    let previous = false;
    return () => {
        let y1;
        let x1;
        let x2;
        let w;
        if (previous) {
            y1 = y2;
            previous = false;
        } else {
            do {
                x1 = randomFloat(-1, 1);
                x2 = randomFloat(-1, 1);
                w = (x1 * x1) + (x2 * x2);
            } while (w >= 1);
            w = Math.sqrt((-2 * Math.log(w)) / w);
            y1 = x1 * w;
            y2 = x2 * w;
            previous = true;
        }
        return (y1 * sd) + mean;
    }
};

// Returns a random sample (either with or without replacement) of size k from an array
const randomSample = <T>(arr: T[], k: number, withReplacement = false): T[] => {
    if (withReplacement) {  // sample with replacement
        return Array.from({length: k}, () => arr[Math.floor(Math.random() * arr.length)]);
    } else { // sample without replacement
        if (k > arr.length) {
            throw new RangeError('Sample size must be less than or equal to array length when sampling without replacement.')
        }
        return arr.map<[T, number]>(a => [a, Math.random()]).sort((a, b) => {
            return a[1] < b[1] ? -1 : 1;
        }).slice(0, k).map(a => a[0]);
    }
};

// Generates a new number if the previously generated exceeds the max or min
// Returned number is between min (inclusive) and max (inclusive)
const withResample = (generate: () => number, min?: number, max?: number) => {
    // hard limit to prevent infinite loops on nonsensical bounds
    for ( let i = 0; i < 100; i++ ) {
        const number = generate();
        // check if out of bounds
        if ( (min !== undefined && number < min) || (max !== undefined && number > max) ) {
            continue;
        }
        return number;
    }
    throw new Error(`Error generating a random number with boundaries min ${min} and max ${max}. Could not find a valid number after 100 attempts.`);
}

export {randomFloat, randomInt, randomGaussian, randomSample};
