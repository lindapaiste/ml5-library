// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Basic Pitch Detection
=== */

let pitch;

async function setup() {
    const audioContext = new AudioContext();
    const stream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
    startPitch(stream, audioContext);
}

setup();

function startPitch(stream, audioContext) {
    pitch = ml5.pitchDetection('./model/', audioContext, stream, modelLoaded);
}

function modelLoaded() {
    document.querySelector('#status').textContent = 'Model Loaded';
    getPitch();
}

function getPitch() {
    pitch.getPitch((err, frequency) => {
        if (frequency) {
            document.querySelector('#result').textContent = frequency;
        } else {
            document.querySelector('#result').textContent = 'No pitch detected';
        }
        getPitch();
    })
}

