// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
KNN Classification on Webcam Images with poseNet. Built with p5.js
=== */
let video;
// Create a KNN classifier
const knnClassifier = ml5.KNNClassifier();
let poseNet;
let poses = [];
let canvas;
const width = 640;
const height = 480;
let ctx;

async function setup() {
    canvas = document.querySelector("#myCanvas");
    ctx = canvas.getContext("2d");

    video = await getVideo();

    // Create the UI buttons
    createButtons();

    // Create a new poseNet method with a single detection
    poseNet = ml5.poseNet(video, modelReady);
    // This sets up an event that fills the global variable "poses"
    // with an array every time new poses are detected
    poseNet.on("pose", results => {
        poses = results;
    });

    requestAnimationFrame(draw);
}

setup();

function draw() {
    requestAnimationFrame(draw);

    ctx.drawImage(video, 0, 0, width, height);

    // We can call both functions to draw all keypoints and the skeletons
    drawKeypoints();
    drawSkeleton();
}

function modelReady() {
    document.querySelector("#status").textContent = "model Loaded";
}

// Add the current frame from the video to the classifier
function addExample(label) {
    // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
    const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);

    // Add an example with a label to the classifier
    knnClassifier.addExample(poseArray, label);
    updateCounts();
}

// Predict the current frame.
function classify() {
    // Get the total number of labels from knnClassifier
    const numLabels = knnClassifier.getNumLabels();
    if (numLabels <= 0) {
        console.error("There is no examples in any label");
        return;
    }
    // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
    const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);

    // Use knnClassifier to classify which label do these features belong to
    // You can pass in a callback function `gotResults` to knnClassifier.classify function
    knnClassifier.classify(poseArray, gotResults);
}

// A util function to create UI buttons
function createButtons() {
    // When the A button is pressed, add the current frame
    // from the video with a label of "A" to the classifier
    document.querySelector("#addClassA")
        .addEventListener("click", () => {
            addExample("A");
        });

    // When the B button is pressed, add the current frame
    // from the video with a label of "B" to the classifier
    document.querySelector("#addClassB")
        .addEventListener("click", () => {
            addExample("B");
        });

    // Reset buttons
    document.querySelector("#resetA")
        .addEventListener("click", () => {
            clearLabel("A");
        });

    document.querySelector("#resetB")
        .addEventListener("click", () => {
            clearLabel("B");
        });

    // Predict button
    document.querySelector("#buttonPredict")
        .addEventListener("click", classify);

    // Clear all classes button
    document.querySelector("#clearAll")
        .addEventListener("click", clearAllLabels);
}

// Show the results
function gotResults(err, result) {
    // Display any error
    if (err) {
        console.error(err);
    }

    if (result.confidencesByLabel) {
        const confidences = result.confidencesByLabel;
        // result.label is the label that has the highest confidence
        if (result.label) {
            document.querySelector("#result").textContent = result.label;
            document.querySelector("#confidence").textContent = `${confidences[result.label] * 100} %`;
        }

        document.querySelector("#confidenceA").textContent = `${
            confidences.A ? confidences.A * 100 : 0
        } %`;
        document.querySelector("#confidenceB").textContent = `${
            confidences.B ? confidences.B * 100 : 0
        } %`;
    }

    classify();
}

// Update the example count for each label
function updateCounts() {
    const counts = knnClassifier.getCountByLabel();

    document.querySelector("#exampleA").textContent = counts.A || 0;
    document.querySelector("#exampleB").textContent = counts.B || 0;
}

// Clear the examples in one label
function clearLabel(classLabel) {
    knnClassifier.clearLabel(classLabel);
    updateCounts();
}

// Clear all the examples in all labels
function clearAllLabels() {
    knnClassifier.clearAllLabels();
    updateCounts();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
    // Loop through all the poses detected
    poses.forEach(pose => {
        // For each pose detected, loop through all the keypoints
        // A keypoint is an object describing a body part (like rightArm or leftShoulder)
        pose.pose.keypoints.forEach(keypoint => {
            // Only draw an ellipse is the pose probability is bigger than 0.2
            if (keypoint.score > 0.2) {
                ctx.fillStyle = "rgb(213, 0, 143)";
                ctx.beginPath();
                ctx.arc(keypoint.position.x, keypoint.position.y, 10, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
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
            ctx.strokeStyle = "#FF0000";
            ctx.stroke();
        });
    });
}

// Helper Functions
async function getVideo() {
    // Grab elements, create settings, etc.
    const videoElement = document.createElement("video");
    videoElement.setAttribute("style", "display: none;");
    videoElement.width = width;
    videoElement.height = height;
    document.body.appendChild(videoElement);

    // Create a webcam capture
    videoElement.srcObject = await navigator.mediaDevices.getUserMedia({video: true});
    await videoElement.play();

    return videoElement;
}
