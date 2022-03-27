// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Real time Object Detection using YOLO
=== */

let yolo;
let objects = [];
let canvas;
let ctx;
const width = 640;
const height = 420;
let img;

async function make() {
  img = new Image();
  img.src = 'images/cat2.JPG';
  img.width = width;
  img.height = height;

  yolo = await ml5.YOLO(startDetecting)

  canvas = createCanvas(width, height);
  ctx = canvas.getContext('2d');
}

// when the dom is loaded, call make();
window.addEventListener('DOMContentLoaded', make);

function startDetecting(){
  console.log('model ready')
  detect();
}

function detect() {
  yolo.detect(img, (err, results) => {
    if(err){
      console.log(err);
      return
    }
    objects = results;

    if(objects){
      draw();
    }
  });
}

function draw(){
  // Clear part of the canvas
  ctx.fillStyle = "#000000"
  ctx.fillRect(0,0, width, height);

  ctx.drawImage(img, 0, 0);
  for (let i = 0; i < objects.length; i += 1) {
      
    ctx.font = "16px Arial";
    ctx.fillStyle = "green";
    ctx.fillText(objects[i].label, objects[i].x + 4, objects[i].y + 16); 

    ctx.beginPath();
    ctx.rect(objects[i].x, objects[i].y, objects[i].width, objects[i].height);
    ctx.strokeStyle = "green";
    ctx.stroke();
    ctx.closePath();
  }
}


function createCanvas(w, h){
  const canvasElement = document.createElement("canvas");
  canvasElement.width  = w;
  canvasElement.height = h;
  document.body.appendChild(canvasElement);
  return canvasElement;
}