// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Multiple Image classification using MobileNet
=== */

// Initialize the Image Classifier method using MobileNet
let classifier;
let allImages = [];
const predictions = [];
let ctx;

async function setup() {
  classifier = await ml5.imageClassifier("MobileNet");
  const res = await fetch("assets/data.json");
  const data = await res.json();

  const canvas = document.querySelector("#myCanvas");
  ctx = canvas.getContext("2d");

  ctx.beginPath();
  ctx.fillStyle = "#eee";
  ctx.rect(0, 0, canvas.width, canvas.height);
  ctx.fill();

  allImages = data.images;

  await Promise.all(allImages.map(async (imgPath, idx) => loadImage(imgPath, idx)));
}
setup();

async function loadImage(imgPath, idx) {
  const imgEl = new Image();
  imgEl.src = `images/dataset/${imgPath}`;

  imgEl.onload = async () => {
    ctx.drawImage(imgEl, 0, 0, imgEl.width, imgEl.height);

    await classifier.classify(imgEl, (err, res) => {
      if (err) {
        console.log(err, idx);
        return;
      }
      const information = {
        name: imgPath,
        result: res,
      };

      predictions.push(information);

      if (predictions.length === allImages.length) {
        console.log(predictions);
        savePredictions();
      } else {
        console.log(information);
      }
    });
  };
}

function download(content, fileName, contentType) {
  const a = document.createElement("a");
  const file = new Blob([content], {
    type: contentType,
  });
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
}

function savePredictions() {
  const predictionsJSON = {
    predictions,
  };
  download(JSON.stringify(predictionsJSON), "json.txt", "text/plain");
}
