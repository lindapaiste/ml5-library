/**
 * Check if a URL string begins with a host scheme
 * @param url
 */
export const isAbsoluteURL = (url: string): boolean => {
  const pattern = new RegExp('^(?:[a-z]+:)?//', 'i');
  return pattern.test(url);
}

/**
 * Turn a relative URL into an absolute URL by pre-pending the current window location
 * @param url
 */
export const getModelPath = (url: string) => {
  return isAbsoluteURL(url) ? url : window.location.pathname + url;
}


/**
 * Creates a string that can be used in a file name.
 * Format is YYYY-MM-DD_mm-hh => "2021-05-02_19-21-42"
 */
export const createDateFileName = (): string => {
  const today = new Date();
  today.setMinutes(today.getMinutes() - today.getTimezoneOffset()); // need to change timezone when using ISO
  const [date, time] = today.toISOString().split(/[T.]/);
  return `${date}_${time.replaceAll(":", "-")}`;
}
/* previous code
    const today = new Date();
    const date = `${today.getFullYear()}-${today.getMonth() + 1}-${today.getDate()}`;
    const time = `${today.getHours()}-${today.getMinutes()}-${today.getSeconds()}`;
    return `${date}_${time}`;
}*/

/**
 * Extract the directory portion from a URL.
 * @param url
 * @param includeTrailingSlash
 */
export const directory = (url: string, includeTrailingSlash = true): string => {
  const noSlash = url.substring(0, url.lastIndexOf('/'));
  return includeTrailingSlash ? noSlash + "/" : noSlash;
}

/**
 * Check that a variable is a FileList, with a guard that makes sure FileList is defined
 * to prevent errors in Node.
 */
export const isFileList = (value: any): value is FileList => {
  return ( FileList !== undefined && value instanceof FileList );
}

/**
 * Helper for creating multiple file paths in a single directory
 */
export class ModelLoader {

  public readonly providedUrl: string;

  /**
   * Create a ModelLoader instance from a URL.
   *
   * @param url - the URL to use as the basis for the loader.
   * Can be a directory or a file path.
   * Can be absolute or relative.
   */
  constructor(url: string) {
    this.providedUrl = url;
  }


  /**
   * Check if the provided URL is absolute or relative
   */
  isAbsolute(): boolean {
    return isAbsoluteURL(this.providedUrl);
  }

  /**
   * Get the absolute URL for the provided URL
   */
  getAbsolutePath(): string {
    return getModelPath(this.providedUrl);
  }

  /**
   * Return either the provided url or the absolute path, depending on the value of the boolean argument.
   *
   * @param absolute
   */
  getPath(absolute = false): string {
    return absolute ? this.getAbsolutePath() : this.providedUrl;
  }

  /**
   * Get the directory for the provided URL. If the provided URL is itself a directory, just return it.
   *
   * @param includeTrailingSlash - Whether to end the directory with "/" or not.
   * @param absolute - Whether to make the directory path absolute (based on the window URL), if it's not already.
   */
  getDirectory(includeTrailingSlash = true, absolute = true): string {
    return directory(this.getPath(absolute), includeTrailingSlash);
  }

  /**
   * Create a path for another file in the same directory as the provided URL.
   * For example, can provide a "model.json" path and get a "metadata.json" path in the same folder.
   *
   * @param fileName - The name of the new file. Should include the extension. Ie. "model.json", "metadata.json".
   * @param absolute - Whether to make the new file path absolute (based on the window URL), if it's not already.
   */
  fileInDirectory(fileName: string, absolute = true): string {
    const directory = this.getDirectory(true, absolute);
    return directory + fileName;
  }

  /**
   * If the provided URL is a path to a ".json" file, return that file.
   * Otherwise assume that it is a directory and return a path to "model.json" in that directory.
   */
  modelJsonPath(absolute = true): string {
    return this.endsWith(".json") ? this.getPath(absolute) : this.fileInDirectory("model.json", absolute);
  }

  /**
   * Helper for checking if the provided URL has a particular ending, such as ".json".
   * Case-insensitive.
   *
   * @param ending - The ending text to check for.
   */
  endsWith(ending: string): boolean {
    return this.providedUrl.toLowerCase().endsWith(ending.toLowerCase());
  }

}

const modelLoader = (url: string) => new ModelLoader(url);

export default modelLoader;