let faceapi;
let video;

// by default all options are set to true
const detectionOptions = {
  withLandmarks: true,
  withDescriptors: false,
};

function setup() {
  createCanvas(360, 270);

  // load up your video
  video = createCapture(VIDEO);
  video.size(width, height);
  // video.hide(); // Hide the video element, and just show the canvas

  faceapi = ml5.faceApi(video, detectionOptions, modelReady);
  textAlign(RIGHT);
}

function modelReady() {
  console.log("ready!");
  console.log(faceapi);
  faceapi.detect(gotResults);
}

function gotResults(err, result) {
  if (err) {
    console.log(err);
    return;
  }
  // console.log(result)
  const detections = result;

  // background(220);
  background(255);
  image(video, 0, 0, width, height);
  if (detections) {
    if (detections.length > 0) {
      // console.log(detections)
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

    noFill();
    stroke(161, 95, 251);
    strokeWeight(2);
    rect(box.x, box.y, box.width, box.height);
  });
}

function drawLandmarks(detections) {
  noFill();
  stroke(161, 95, 251);
  strokeWeight(2);

  detections.forEach(detection => {
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
  beginShape();
  feature.forEach(point => {
    vertex(point.x, point.y);
  });

  if (closed === true) {
    endShape(CLOSE);
  } else {
    endShape();
  }
}
