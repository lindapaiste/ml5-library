// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */

let poseNet;
let poses = [];

let video;
let canvas;
let ctx;

async function setup() {
    // Grab elements, create settings, etc.
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    video.srcObject = await navigator.mediaDevices.getUserMedia({video: true});
    video.play();
    // Create a new poseNet method with a single detection
    poseNet = await ml5.poseNet(video, modelReady);
    // This sets up an event that fills the global variable "poses"
    // with an array every time new poses are detected
    poseNet.on('pose', results => {
        poses = results;
    });

    requestAnimationFrame(draw);
}

setup();

function modelReady() {
    console.log('model loaded!')
}

// Draw an ellipse around a given point
function drawArc(point) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 10, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
}

function draw() {
    requestAnimationFrame(draw);

    ctx.drawImage(video, 0, 0, 640, 480);
    // We can call both functions to draw all keypoints and the skeletons

    // For one pose only (use a for loop for multiple poses!)
    if (poses.length > 0) {
        const {pose} = poses[0];

        // Create a pink ellipse for the nose
        ctx.fillStyle = 'rgb(213, 0, 143)';
        drawArc(pose.nose);

        // Create a yellow ellipse for the right eye
        ctx.fillStyle = 'rgb(255, 215, 0)';
        drawArc(pose.rightEye);

        // Create a yellow ellipse for the right eye
        ctx.fillStyle = 'rgb(255, 215, 0)';
        drawArc(pose.leftEye);
    }
}
