import createTextToxicity from "./index";

describe("TextToxicity", () => {
    it("classifies text", async () => {
        const model = createTextToxicity();
        // TODO: figure out a better way to get the types
        if (!model.classify) return;
        const res = await model.classify([
            'We\'re dudes on computers, moron. You are quite astonishingly stupid.',
        'Please stop. If you continue to vandalize Wikipedia, as you did to Kmart, you will be blocked from editing.'] as any);
        expect(res).toEqual(["s"]);
    })
})
