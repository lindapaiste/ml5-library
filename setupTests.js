// eslint-disable-next-line import/no-extraneous-dependencies
const { ImageData } = require('canvas');
// eslint-disable-next-line import/no-extraneous-dependencies
const { TextEncoder, TextDecoder } = require('util');
// eslint-disable-next-line import/no-extraneous-dependencies
require('jsdom-global')();
// require('@tensorflow/tfjs-node');

async function setupTests() {
  console.log("Beginning setup");

  // Use the node-canvas ImageData polyfill
  if (!global.ImageData) {
    global.ImageData = ImageData;
  }

  if (!global.TextEncoder) {
    global.TextEncoder = TextEncoder;
  }

  if (!global.TextDecoder) {
    global.TextDecoder = TextDecoder;
  }

  console.log(global.ImageData, global.TextEncoder, global.TextDecoder)

  console.log("Setup complete");
}

module.exports = setupTests;
