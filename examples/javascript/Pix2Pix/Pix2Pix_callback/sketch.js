// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Pix2pix Edges2Pikachu example with p5.js using callback functions
This uses a pre-trained model on Pikachu images
For more models see: https://github.com/ml5js/ml5-data-and-training/tree/master/models/pix2pix
=== */

// The pre-trained Edges2Pikachu model is trained on 256x256 images
// So the input images can only be 256x256 or 512x512, or multiple of 256
const SIZE = 256;
let inputImg;
let outputContainer;
let statusMsg;
let pix2pix;
let clearBtn;
let transferBtn;
let modelReady = false;
let isTransferring = false;
let outputImg;

let pX = null;
let pY = null;
let mouseX = null;
let mouseY = null;
let mouseDown = false;

const canvasElement = document.createElement("canvas");
const ctx = canvasElement.getContext("2d");

async function setup() {
  // Create a canvas
  createCanvas(SIZE, SIZE);

  // Display initial input image
  inputImg = document.querySelector("#inputImage");
  outputImg = document.querySelector('#outputImage');
  drawImage();

  // Select output div container
  outputContainer = document.querySelector('#output');
  statusMsg = document.querySelector('#status');

  // Select 'transfer' button html element
  transferBtn = document.querySelector('#transferBtn');

  // Select 'clear' button html element
  clearBtn = document.querySelector('#clearBtn');


  // Attach a mousePressed event to the 'clear' button
  clearBtn.addEventListener('click', ()=> {
    clearCanvas();
  });


  canvasElement.addEventListener('mousemove', onMouseUpdate);
  canvasElement.addEventListener('mousedown', onMouseDown);
  canvasElement.addEventListener('mouseup', onMouseUp);


  // Create a pix2pix method with a pre-trained model
  pix2pix = await ml5.pix2pix('models/edges2pikachu.pict', modelLoaded);

  requestAnimationFrame(draw)
  // drawImage();
}

setup();

// Draw on the canvas when mouse is pressed
function draw() {
  requestAnimationFrame(draw);

  if (pX == null || pY == null) {
    pX = mouseX;
    pY = mouseY;
    drawImage();
  }

  if(mouseDown){
    // Set stroke weight to 10
    ctx.lineWidth = 10;
    // Set stroke color to black
    ctx.strokeStyle = "#000000";
    // If mouse is pressed, draw line between previous and current mouse positions
    ctx.beginPath();
    ctx.lineCap = "round";
    ctx.moveTo(mouseX, mouseY);
    ctx.lineTo(pX, pY);
    ctx.stroke();
  }
  

  pX = mouseX;
  pY = mouseY;
}

// Whenever mouse is released, transfer the current image if the model is loaded and it's not in the process of another transformation
function mouseReleased() {
  if (modelReady && !isTransferring) {
    transfer()
  }
}

// A function to be called when the models have loaded
function modelLoaded() {
  // Show 'Model Loaded!' message
  statusMsg.textContent = 'Model Loaded!';

  // Set modelReady to true
  modelReady = true;

  // Call transfer function after the model is loaded
  transfer();

  // Attach a mousePressed event to the transfer button
  transferBtn.addEventListener('click',() => {
    transfer();
  });
}

// Draw the input image to the canvas
function drawImage() {
  ctx.drawImage(inputImg, 0,0, SIZE, SIZE);
}

// Clear the canvas
function clearCanvas() {
  ctx.fillStyle = '#ebedef'
  ctx.fillRect(0, 0, SIZE, SIZE);
}

function transfer() {
  // Set isTransferring to true
  isTransferring = true;

  // Update status message
  statusMsg.textContent = 'Applying Style Transfer...!';

  // Apply pix2pix transformation
  pix2pix.transfer(canvasElement, (err, result) => {
    if (err) {
      console.log(err);
    }
    if (result && result.src) {
      // Set isTransferring back to false
      isTransferring = false;
      // Clear output container
      outputContainer.innerHTML = '';
      // Create an image based result
      // createImg(result.src).class('border-box').parent('output');
      outputImg.src =  result.src;

      console.log(result);
      // Show 'Done!' message
      statusMsg.textContent = 'Done!';
    }
  });
}


function createCanvas(w, h) {
  canvasElement.width = w;
  canvasElement.height = h;
  document.body.appendChild(canvasElement);
  ctx.fillStyle = '#ebedef'
  ctx.fillRect(0, 0, w, h);
}


function onMouseDown() {
  mouseDown = true;
}

function onMouseUp() {
  mouseDown = false;
}

function onMouseUpdate(e) {
  const rect = canvasElement.getBoundingClientRect();
  mouseX = e.clientX - rect.left;
  mouseY = e.clientY - rect.top;
}