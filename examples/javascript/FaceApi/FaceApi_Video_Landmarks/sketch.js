let faceapi;
let video;
const width = 360;
const height = 280;
let canvas;
let ctx;

// by default all options are set to true
const detectionOptions = {
  withLandmarks: true,
  withDescriptors: false,
};

async function make() {
  // get the video
  video = await getVideo();

  canvas = createCanvas(width, height);
  ctx = canvas.getContext("2d");

  faceapi = ml5.faceApi(video, detectionOptions, modelReady);
}
// call app.map.init() once the DOM is loaded
window.addEventListener("DOMContentLoaded", make);

function modelReady() {
  console.log("ready!");
  faceapi.detect(gotResults);
}

function gotResults(err, result) {
  if (err) {
    console.log(err);
    return;
  }

  // console.log(result)
  const detections = result;

  // Clear part of the canvas
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, width, height);

  ctx.drawImage(video, 0, 0, width, height);

  if (detections) {
    if (detections.length > 0) {
      drawBox(detections);
      drawLandmarks(detections);
    }
  }
  faceapi.detect(gotResults);
}

function drawBox(detections) {
  detections.forEach(detection => {
    // eslint-disable-next-line prefer-destructuring
    const box = detection.alignedRect.box;

    ctx.beginPath();
    ctx.rect(box.x, box.y, box.width, box.height);
    ctx.strokeStyle = "#a15ffb";
    ctx.stroke();
    ctx.closePath();
  });
}

function drawLandmarks(detections) {
  detections.forEach( detection => {
    const {mouth, nose, leftEye, rightEye, leftEyeBrow, rightEyeBrow} = detection.parts;

    drawPart(mouth, true);
    drawPart(nose, false);
    drawPart(leftEye, true);
    drawPart(leftEyeBrow, false);
    drawPart(rightEye, true);
    drawPart(rightEyeBrow, false);

  });
}

function drawPart(feature, closed) {
  ctx.beginPath();
  feature.forEach((point, i) => {
    if (i === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });

  if (closed === true) {
    ctx.closePath();
  }
  ctx.stroke();
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

function createCanvas(w, h){
  const canvasElement = document.createElement("canvas");
  canvasElement.width  = w;
  canvasElement.height = h;
  document.body.appendChild(canvasElement);
  return canvasElement;
}