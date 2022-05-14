// eslint-disable-next-line global-require,import/no-extraneous-dependencies
const { TestEnvironment } = require('jest-environment-node');

module.exports = class CustomTestEnvironment extends TestEnvironment {


  constructor(config, context) {
    console.log(config, context);
    super(config, context);
  }
  async setup() {
    await super.setup();
    if (typeof this.global.TextEncoder === 'undefined') {
      // eslint-disable-next-line global-require
      const { TextEncoder, TextDecoder } = require('util');
      this.global.TextEncoder = TextEncoder;
      this.global.TextDecoder = TextDecoder;
    }
    console.log("custom env");
  }
}
