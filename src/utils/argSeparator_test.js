import {ArgSeparator} from "./argSeparator";

describe("ArgSeparator", () => {

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

        // partially filled args
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

        // completely filled args
        const d = new ArgSeparator(video, options, callback, "modelName", 99);
        expect(d.string).toBe("modelName");
        expect(d.number).toBe(99);
        expect(d.video).toBe(video);
        expect(d.image).toBe(video);
        expect(d.options).toBe(options);
        expect(d.callback).toBe(callback);
    });

    it("can differentiate between image and video", () => {
        const imageOnly = new ArgSeparator(image);
        expect(imageOnly.video).toBeUndefined();
        expect(imageOnly.image).toBe(image);

        const videoOnly = new ArgSeparator(video);
        expect(videoOnly.video).toBe(video);
        expect(videoOnly.image).toBe(image);
    });

    it("can override earlier args with later ones", () => {
        const numbers = new ArgSeparator(1,2,3);
        expect(numbers.number).toBe(3);

        const imageAndVideo = new ArgSeparator(video, image);
        expect(imageAndVideo.video).toBe(video);
        expect(imageAndVideo.image).toBe(image);
    });

    it("can require arguments and throw errors", () => {
        const missingImage = new ArgSeparator(callback, options);
        expect(missingImage.require('image')).toThrowError();
    })


    // note: wanted to test that it would throw an error on an invalid element like `div`,
    // but it won't because it is assignable to `options`.

})