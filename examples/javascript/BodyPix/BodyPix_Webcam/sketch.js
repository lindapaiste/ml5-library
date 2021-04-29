let bodypix;
let video;
let ctx;
const width = 480;
const height = 360;

const options = {
  outputStride: 8, // 8, 16, or 32, default is 16
  segmentationThreshold: 0.3 // 0 - 1, defaults to 0.5 
}

async function setup() {
  // create a canvas to draw to
  const canvas = createCanvas();
  document.body.appendChild(canvas);
  ctx = canvas.getContext('2d');
  // get the video
  video = await getVideo();
  // load bodyPix with video
  bodypix = await ml5.bodyPix(options)
}

// when the dom is loaded, call make();
window.addEventListener('DOMContentLoaded', setup);

function videoReady() {
  // run the segmentation on the video, handle the results in a callback
  bodypix.segment(video, gotImage, options);
}

function gotImage(err, result){
  if(err) {
    console.log(err);
    return;
  }
  // draw the video image to the canvas
  ctx.drawImage(video, 0, 0, width, height);

  // convert the mask into a black and white image
  const maskedBackground = imageDataToCanvas(result.raw.backgroundMask);

  // draw the mask on top of the video
  ctx.drawImage(maskedBackground, 0, 0);
    
  bodypix.segment(video, gotImage, options);
}

// Helper Functions
async function getVideo(){
  // Grab elements, create settings, etc.
  const videoElement = document.createElement('video');
  videoElement.setAttribute("style", "display: none;"); 
  videoElement.width = width;
  videoElement.height = height;
  videoElement.onloadeddata = videoReady;
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