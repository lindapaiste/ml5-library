// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Image and Video base class
*/

import { getImageElement, VideoArg } from "../handleArguments";

class Video {

    videoReady: boolean;
    size: number;
    videoElt?: HTMLMediaElement;
    video?: HTMLVideoElement;

    constructor(video: VideoArg | undefined, size: number) {
        this.videoElt = getImageElement<HTMLVideoElement>(video);
        this.size = size;
        this.videoReady = false;
    }

    async loadVideo(): Promise<HTMLVideoElement> {
        if (!this.videoElt) {
            throw new Error("No video provided");
        }
        this.video = document.createElement('video');
        let stream: MediaStream
        const sUsrAg = navigator.userAgent;
        if (sUsrAg.indexOf('Firefox') > -1) {
            // @ts-ignore
            stream = this.videoElt.mozCaptureStream();
        } else {
            // @ts-ignore
            stream = this.videoElt.captureStream();
        }
        this.video.srcObject = stream;
        this.video.width = this.size;
        this.video.height = this.size;
        this.video.autoplay = true;
        this.video.playsInline = true;
        this.video.muted = true;
        await this.video.play();
        return this.video;
    }
}

export default Video;
