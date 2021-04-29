let faceapi;
let img;
const width = 200;
const height = 200;
let canvas;
let ctx;

// by default all options are set to true
const detectionOptions = {
    withLandmarks: true,
    withDescriptors: false,
};

async function make() {
    img = new Image();
    img.src = "assets/frida.jpg";
    img.width = width;
    img.height = height;

    canvas = createCanvas(width, height);
    ctx = canvas.getContext("2d");

    faceapi = await ml5.faceApi(detectionOptions, modelReady);

    // faceapi.detectSingle(img, gotResults)
}

// call app.map.init() once the DOM is loaded
window.addEventListener("DOMContentLoaded", make);

function modelReady() {
    console.log("ready!");
    faceapi.detectSingle(img, gotResults);
}

function gotResults(err, result) {
    if (err) {
        console.log(err);
        return;
    }
    // console.log(result)
    const detections = result;

    ctx.drawImage(img, 0, 0, width, height);

    if (detections) {
        console.log(detections);
        drawBox(detections);
        drawLandmarks(detections);
    }
}

function drawBox(detections) {
    const {alignedRect} = detections;
    const {_x, _y, _width, _height} = alignedRect._box;
    // canvas.fillStyle = 'none';
    ctx.rect(_x, _y, _width, _height);
    ctx.strokeStyle = "#a15ffb";
    ctx.stroke();
}

function drawLandmarks(detections) {
    const {mouth, nose, leftEye, rightEye, leftEyeBrow, rightEyeBrow} = detections.parts;

    drawPart(mouth, true);
    drawPart(nose, false);
    drawPart(leftEye, true);
    drawPart(leftEyeBrow, false);
    drawPart(rightEye, true);
    drawPart(rightEyeBrow, false);
}

function drawPart(feature, closed) {
    ctx.beginPath();
    feature.forEach((point, i) => {
        const {_x, _y} = point;

        if (i === 0) {
            ctx.moveTo(_x, _y);
        } else {
            ctx.lineTo(_x, _y);
        }
    });

    if (closed === true) {
        ctx.closePath();
    }
    ctx.stroke();
}

// Helper Functions
function createCanvas(w, h){
    const canvasElement = document.createElement("canvas");
    canvasElement.width  = w;
    canvasElement.height = h;
    document.body.appendChild(canvasElement);
    return canvasElement;
}