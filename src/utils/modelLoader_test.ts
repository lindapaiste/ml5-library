import modelLoader, {createDateFileName} from "./modelLoader";
// TODO: how to import utilities which are not exported in the root ml5 package?

describe("ModelLoader utility functions", () => {

    it("can create a date-based file name", () => {
        // Uses current timestamp, so can't check the exact value.  Instead check that it matches a regex.
        const fileName = createDateFileName();
        const regex = new RegExp(/^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$/);
        // Verify that the regex is correct by checking against an expected output
        expect(regex.test( "2021-05-02_19-21-42")).toBeTrue();
        // Check the created file name against the regex
        expect(regex.test( fileName)).toBeTrue();
    });
})

describe("ModelLoader class", () => {

    const absoluteUrl = 'https://example.com/a/b/c/folder/model.json';
    const relativeUrl = '/folder/model.json';
    const directoryUrl = 'https://example.com/a/b/c/folder';

    const absoluteLoader = modelLoader(absoluteUrl);
    const relativeLoader = modelLoader(relativeUrl);
    const directoryLoader = modelLoader(directoryUrl);

    it("Can extract a directory from a file", () => {
        expect(absoluteLoader.getDirectory(true)).toBe('https://example.com/a/b/c/folder/');
        expect(absoluteLoader.getDirectory(false)).toBe('https://example.com/a/b/c/folder');
        expect(relativeLoader.getDirectory(true)).toBe('/folder/');
        expect(relativeLoader.getDirectory(false)).toBe('/folder');
        expect(directoryLoader.getDirectory(true)).toBe('https://example.com/a/b/c/folder/');
        expect(directoryLoader.getDirectory(false)).toBe('https://example.com/a/b/c/folder');
    });

    it("Can turn a relative URL to absolute", () => {
        // TODO: how to test this properly?  It depends on window.location.
        expect(relativeLoader.getAbsolutePath().startsWith("http")).toBeTrue();
        // Absolute path is unchanged
        expect(absoluteLoader.getAbsolutePath()).toBe(absoluteUrl);
    });

    it("Can create a path for another file in the same folder", () => {
       expect(absoluteLoader.fileInDirectory('metadata.json')).toBe('https://example.com/a/b/c/folder/metadata.json');
       expect(relativeLoader.fileInDirectory('metadata.json')).toBe('/folder/metadata.json');
       expect(directoryLoader.fileInDirectory('metadata.json')).toBe('https://example.com/a/b/c/folder/metadata.json');
    });
})