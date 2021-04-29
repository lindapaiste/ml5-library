import {ArgSeparator} from "./argSeparator";

describe("ImageModelArgs", () => {

    const video = new HTMLVideoElement();
    const image = new HTMLImageElement();
    // const canvas = new HTMLCanvasElement();
    // const data = new ImageData(0, 0);
   // const tensor = new Tensor();

    const options = {
        multiplier: 0.75,
        outputStride: 16,
        segmentationThreshold: 0.5,
    }

    const callback = () => {}

    it("can separate arguments", () => {
        const a = new ArgSeparator(video, callback);
        expect(a.options).toBeUndefined();
        expect(a.video).toBe(video);
        expect(a.callback).toBe(callback);
        expect(a.image).toBe(video);

        const b = new ArgSeparator(options);
        expect(b.video).toBeUndefined();
        expect(b.options).toBe(options);
        expect(b.callback).toBeUndefined();
        expect(b.image).toBeUndefined();

        const c = new ArgSeparator(image);
        expect(c.video).toBeUndefined();
        expect(c.image).toBe(image);
    });

    // note: wanted to test that it would throw an error on an invalid element like `div`,
    // but it won't because it is assignable to `options`.

})