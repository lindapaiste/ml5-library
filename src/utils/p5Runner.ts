import * as p5 from "p5";
// This just imports the types, not the objects because p5 is not a dependency
// Only use for type annotations
// TODO: actual p5 defs are in p5/global.d.ts
export const runP5Function = <T>(f: (p5: p5) => T): T => {

}