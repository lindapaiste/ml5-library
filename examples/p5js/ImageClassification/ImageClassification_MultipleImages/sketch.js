// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Multiple Image classification using MobileNet and p5.js
=== */


let classifier;
let currentIndex = 0;
const allImages = [];
const predictions = [];
let data;

// Initialize the Image Classifier method using MobileNet
function preload() {
  classifier = ml5.imageClassifier("MobileNet");
  data = loadJSON("assets/data.json");
}

function setup() {
  createCanvas(400, 400);
  background(0);
  appendImages();
  loadImage(allImages[currentIndex], imageReady);
}

function appendImages() {
  data.images.forEach(imgPath => {
    allImages.push(`images/dataset/${imgPath}`);
  });
}

// When the image has been loaded,
// get a prediction for that image
function imageReady(img) {
  image(img, 0, 0);
  classifier.classify(img, gotResult);
}

function savePredictions() {
  saveJSON({ predictions }, "predictions.json");
}

// When we get the results
function gotResult(err, results) {
  // If there is an error, show in the console
  if (err) {
    console.error(err);
    return;
  }

  const information = {
    name: allImages[currentIndex],
    result: results,
  };

  predictions.push(information);
  createDiv(`Label: ${results[0].label}`);
  createDiv(`Confidence: ${nf(results[0].confidence, 0, 2)}`);
  currentIndex += 1;
  if (currentIndex <= allImages.length - 1) {
    loadImage(allImages[currentIndex], imageReady);
  } else {
    savePredictions();
  }
}
