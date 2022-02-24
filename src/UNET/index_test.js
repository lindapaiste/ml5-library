// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import {asyncLoadImage, getRobin} from '../utils/testingUtils';
import uNet from './index';

// test image
function returnTestImageData (){
  const r = "data:image/jpg;base64,/9j/4QAYRXhpZgAASUkqAAgAAAAAAAAAAAAAAP/sABFEdWNreQABAAQAAAABAAD/4QMfaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/PiA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJBZG9iZSBYTVAgQ29yZSA1LjYtYzE0NSA3OS4xNjM0OTksIDIwMTgvMDgvMTMtMTY6NDA6MjIgICAgICAgICI+IDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+IDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiIHhtbG5zOnhtcE1NPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvbW0vIiB4bWxuczpzdFJlZj0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL3NUeXBlL1Jlc291cmNlUmVmIyIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bXBNTTpEb2N1bWVudElEPSJ4bXAuZGlkOkQyOEVGRTdFRUVDNzExRTlCQzc3RDc3QUQ4OTlEN0UyIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOkQyOEVGRTdERUVDNzExRTlCQzc3RDc3QUQ4OTlEN0UyIiB4bXA6Q3JlYXRvclRvb2w9IkFkb2JlIFBob3Rvc2hvcCBDQyAyMDE5IE1hY2ludG9zaCI+IDx4bXBNTTpEZXJpdmVkRnJvbSBzdFJlZjppbnN0YW5jZUlEPSI4MTI4MTdEREVDODQ4MUY5RjY0RkRCRDIxODNDQkVFQyIgc3RSZWY6ZG9jdW1lbnRJRD0iODEyODE3RERFQzg0ODFGOUY2NEZEQkQyMTgzQ0JFRUMiLz4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz7/7gAOQWRvYmUAZMAAAAAB/9sAhAAaGRknHCc+JSU+Qi8vL0JHPTs7PUdHR0dHR0dHR0dH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";
  return r
}

// test image is stored in a base64 string so that this script can be self-contained.
// test image is a portrait of a woman, and is royalty free.

describe('UNET', ()=>{
  let model;
  let segmentationImage;

  beforeAll(async () => {
    jest.setTimeout(15000);
    model = await uNet('face');
    const src = returnTestImageData();
    const testImage = await asyncLoadImage(src);
    segmentationImage = await model.segment(testImage);
  });

  it("Model is ready",()=> {
    expect(model.ready).toBeTruthy()
  });

  it("Segmentation image has 16384 pixels", ()=>{
    expect(segmentationImage.raw.backgroundMask.length / 4 ).toBe(16384);
  });

  it("Pixel 0 should be 0 since it's in the background", ()=>{
    expect(segmentationImage.raw.backgroundMask[0]).toBe(0);
    expect(segmentationImage.raw.backgroundMask[1]).toBe(0);
    expect(segmentationImage.raw.backgroundMask[2]).toBe(0);
  });

  it("R channel of Pixel 7496 should be above 200", ()=>{
    // since it's between her eyebrows
    expect(segmentationImage.raw.backgroundMask[7496*4]).toBeGreaterThan(200);
  });
});


