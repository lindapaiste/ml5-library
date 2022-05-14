import { JSDOM } from 'jsdom';
import { waitFor, click } from '@testing-library/dom';

console.log(process.version);
console.log(new TextEncoder());

describe("DCGAN Example", () => {
  it("generates an image", async () => {
    const dom = await(JSDOM.fromFile("index.html", { resources: "usable", runScripts: "dangerously" }));
    expect(dom).toEqual(5);
    expect(dom.serialize()).toEqual(5);
    const document = dom.window.document;
    const button = document.querySelector("button");
    jest.spyOn(dom.window.ml5, "dcgan").and.callThrough();
    await click(button);
    expect(dom.window.ml5.dcgan).toHaveBeenCalledTimes(1);
  })
})
