async function make() {
  const img = new Image();
  img.src = 'data/harriet.jpg'
  const {width, height} = img;

  const bodypix = await ml5.bodyPix()
  const segmentation = await bodypix.segment(img);

  const canvas = createCanvas(width, height);
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // draw the photo to the canvas
  ctx.drawImage(img, 0,0);

  // convert the mask into a black and white image
  const maskedBackground = imageDataToCanvas(segmentation.raw.backgroundMask);

  // draw the mask on top of the photo
  ctx.drawImage(maskedBackground, 0, 0);
}

// call make() once the DOM is loaded
window.addEventListener('DOMContentLoaded', make);

// Convert a ImageData to a Canvas
function imageDataToCanvas(imageData) {
  const {width, height} = imageData;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  return ctx.canvas;
}

function createCanvas(w, h){
  const element = document.createElement("canvas");
  element.width  = w;
  element.height = h;
  return element;
}