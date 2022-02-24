/**
 * Want to move all video interactions to a central place
 */
export default class MediaWrapper {
    // call this elt so that ArgSeparator can pick up on a MediaWrapper
    public readonly elt: HTMLMediaElement;

    //public isReady: boolean = false;

    constructor(element: HTMLMediaElement | { elt: HTMLMediaElement }) {
        // handle p5
        if ("elt" in element) {
            this.elt = element.elt
        } else {
            this.elt = element;
        }
    }

    get isReady() {
        /** explanation of numbers - https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/readyState#value
         *
         HAVE_NOTHING        0    No information is available about the media resource.
         HAVE_METADATA       1    Enough of the media resource has been retrieved that the metadata attributes are initialized. Seeking will no longer raise an exception.
         HAVE_CURRENT_DATA   2    Data is available for the current playback position, but not enough to actually play more than one frame.
         HAVE_FUTURE_DATA    3    Data for the current playback position as well as for at least a little bit of time into the future is available (in other words, at least two frames of video, for example).
         HAVE_ENOUGH_DATA    4    Enough data is available—and the download rate is high enough—that the media can be played through to the end without interruption.
         */
        return this.elt.readyState >= 2;
    }

    async load(): Promise<this> {
        if (this.isReady) {
            return this;
        }

        return new Promise((resolve, reject) => {
            // Fired when the first frame of the media has finished loading.
            this.elt.onloadeddata = () => resolve(this);
            // Fired when the resource could not be loaded due to an error.
            this.elt.onerror = () => reject(new Error(`Error loading media file ${this.elt.src}`));
        });
    }

    // TODO: tf.nextFrame here or in component?

    // TODO: can play, pause, restart, etc as needed

}