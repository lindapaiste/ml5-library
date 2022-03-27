let bodypix;
let video;
let canvas;
let ctx;
const width = 480;
const height = 360;

const options = {
  outputStride: 8, // 8, 16, or 32, default is 16
  segmentationThreshold: 0.3 // 0 - 1, defaults to 0.5 
}

async function make() {
  canvas = createCanvas();
  document.body.appendChild(canvas);
  ctx = canvas.getContext('2d');
  // get the video
  video = await getVideo();
  // load bodyPix with video
  bodypix = await ml5.bodyPix(video)
  // run the segmentation on the video, handle the results in a callback
  bodypix.segmentWithParts(gotImage, options);
}

// when the dom is loaded, call make();
window.addEventListener('DOMContentLoaded', make);


function gotImage(err, result){
  if(err) {
    console.log(err);
    return;
  }
  ctx.drawImage(video, 0, 0, width, height);

  const parts = imageDataToCanvas(result.raw.partMask)
  ctx.drawImage(parts, 0, 0, width, height);

  bodypix.segmentWithParts(gotImage, options);
}

// Helper Functions
async function getVideo(){
  // Grab elements, create settings, etc.
  const videoElement = document.createElement('video');
  videoElement.setAttribute("style", "display: none;"); 
  videoElement.width = width;
  videoElement.height = height;
  document.body.appendChild(videoElement);

  // Create a webcam capture
  videoElement.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
  await videoElement.play();

  return videoElement
}
// Convert a ImageData to a Canvas
function imageDataToCanvas(imageData) {
  const newCanvas = createCanvas();
  const newCtx = newCanvas.getContext('2d');
  newCtx.putImageData(imageData, 0, 0);
  return newCtx.canvas;
}

function createCanvas(){
  const element = document.createElement("canvas");
  element.width  = width;
  element.height = height;
  return element;
}