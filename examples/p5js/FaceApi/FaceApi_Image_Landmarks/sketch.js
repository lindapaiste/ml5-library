let faceapi;
let img;

// by default all options are set to true
const detectionOptions = {
  withLandmarks: true,
  withDescriptors: false,
};

function preload() {
  img = loadImage("assets/frida.jpg");
}

function setup() {
  createCanvas(200, 200);
  img.resize(width, height);

  faceapi = ml5.faceApi(detectionOptions, modelReady);
  textAlign(RIGHT);
}

function modelReady() {
  console.log("ready!");
  console.log(faceapi);
  faceapi.detectSingle(img, gotResults);
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
  image(img, 0, 0, width, height);
  if (detections) {
    // console.log(detections)
    drawBox(detections);
    drawLandmarks(detections);
  }
}

function drawBox(detections) {
  // eslint-disable-next-line prefer-destructuring
  const box = detections.alignedRect.box;

  noFill();
  stroke(161, 95, 251);
  strokeWeight(2);
  rect(box.x, box.y, box.width, box.height);
}

function drawLandmarks(detections) {
  noFill();
  stroke(161, 95, 251);
  strokeWeight(2);

  push();
  // mouth
  beginShape();
  detections.parts.mouth.forEach(item => {
    vertex(item.x, item.y);
  });
  endShape(CLOSE);

  // nose
  beginShape();
  detections.parts.nose.forEach(item => {
    vertex(item.x, item.y);
  });
  endShape(CLOSE);

  // left eye
  beginShape();
  detections.parts.leftEye.forEach(item => {
    vertex(item.x, item.y);
  });
  endShape(CLOSE);

  // right eye
  beginShape();
  detections.parts.rightEye.forEach(item => {
    vertex(item.x, item.y);
  });
  endShape(CLOSE);

  // right eyebrow
  beginShape();
  detections.parts.rightEyeBrow.forEach(item => {
    vertex(item.x, item.y);
  });
  endShape();

  // left eye
  beginShape();
  detections.parts.leftEyeBrow.forEach(item => {
    vertex(item.x, item.y);
  });
  endShape();

  pop();
}
