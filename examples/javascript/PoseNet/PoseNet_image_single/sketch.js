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

let img;
let canvas;
let ctx;

async function setup() {
  // Grab elements, create settings, etc.
  img = document.getElementById("image");

  canvas = document.getElementById("canvas");
  canvas.width = 640;
  canvas.height = 360;
  ctx = canvas.getContext("2d");

  // Create a new poseNet method with a single detection
  poseNet = await ml5.poseNet(modelReady);
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on("pose", results => {
    poses = results;
  });

  requestAnimationFrame(draw);
}

setup();

function modelReady() {
  console.log("model loaded!");
  poseNet.singlePose(img);
}

function draw() {
  requestAnimationFrame(draw);

  ctx.drawImage(img, 0, 0, img.width, img.height);
  // We can call both functions to draw all keypoints and the skeletons

  // For one pose only (use a for loop for multiple poses!)
  if (poses.length > 0) {
    drawKeypoints();
    drawSkeleton();
  }
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  poses.forEach(pose => {
    // For each pose detected, loop through all the keypoints
    // A keypoint is an object describing a body part (like rightArm or leftShoulder)
    const {keypoints} = pose.pose;
    keypoints.forEach(keypoint => {
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        ctx.fillStyle = "#FFFFFF";
        ctx.beginPath();
        ctx.arc(keypoint.position.x, keypoint.position.y, 6, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.strokeStyle = "#00FF00";
        ctx.fill();
      }
    })
  })
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  poses.forEach(pose => {
    const {skeleton} = pose;
    // For every skeleton, loop through all body connections
    skeleton.forEach(part => {
      const [partA, partB] = part;
      ctx.beginPath();
      ctx.moveTo(partA.position.x, partA.position.y);
      ctx.lineTo(partB.position.x, partB.position.y);
      ctx.strokeStyle = "#FFFFFF";
      ctx.stroke();
    });
  });
}