// eslint-disable-next-line import/no-extraneous-dependencies
import { createImageData } from "canvas";
import { isImageData } from "./imageUtilities";
import { getRobin, randomImageData } from "./testingUtils";

describe("imageUtilities", () => {
  const canvas = document.createElement('canvas');
  canvas.width = 200;
  canvas.height = 200;

  describe("isImageData", () => {
    it("Can identify ImageData from a canvas", () => {
      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0,0, 200, 200);
      expect(isImageData(imgData)).toBe(true);
    });
    it("Can identify an ImageData object", async () => {
      const emptyData = createImageData(200, 200);
      expect(isImageData(emptyData)).toBe(true);
      const randomData = await randomImageData(200, 200);
      expect(isImageData(randomData)).toBe(true);
    })
    it("Will return false for other image types", async () => {
      expect(isImageData(canvas)).toBe(false);
      const image = await getRobin();
      expect(isImageData(image)).toBe(false);
    })
  })
})
