// Select elements from the HTML
const rSlider = document.getElementById("red");
const gSlider = document.getElementById("green");
const bSlider = document.getElementById("blue");
const labelP = document.getElementById("label");
const lossP = document.getElementById("loss");
const statusP = document.getElementById("status");
const form = document.getElementById("sliders");
const color = document.getElementById("color");
const random = document.getElementById("random");

// Cannot classify before the model is ready.
let isReady = false;
let isTrained = false;

// Read the current values of the sliders.
function currentColor() {
  return {
    r: parseInt(rSlider.value, 10),
    g: parseInt(gSlider.value, 10),
    b: parseInt(bSlider.value, 10)
  }
}

// Respond to changes in the sliders.
function onChangeColor() {
  const { r, g, b } = currentColor();
  // Make the colored square match the slider values.
  color.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
  // Classify the new color.
  if (isReady) {
    classify();
  }
}
form.addEventListener('change', onChangeColor);

// Set sliders to a random color.
function randomize() {
  rSlider.value = Math.floor(256 * Math.random());
  gSlider.value = Math.floor(256 * Math.random());
  bSlider.value = Math.floor(256 * Math.random());
  onChangeColor();
}
// ...when the button is clicked.
random.addEventListener('click', randomize);

// Initialize the neural network and feed in training data.
const nnOptions = {
  dataUrl: 'data/colorData.json',
  inputs: ['r', 'g', 'b'],
  outputs: ['label'],
  task: 'classification',
  debug: true
};
const neuralNetwork = ml5.neuralNetwork(nnOptions, modelReady);

// Function to call when the neural network has been created.
function modelReady() {
  statusP.innerText = "Training...";
  neuralNetwork.normalizeData();
  isReady = true;
  const trainingOptions = {
    epochs: 20,
    batchSize: 64
  }
  neuralNetwork.train(trainingOptions, whileTraining, finishedTraining);
  // Start guessing while training!
  classify();
}

// Called on each epoch.
function whileTraining(epoch, logs) {
  lossP.innerText = `Epoch: ${epoch} - loss: ${logs.loss.toFixed(2)}`;
}

// Called when training is complete.
function finishedTraining() {
  statusP.innerText = "Training Complete"
  isTrained = true;
}

// Take the color from the sliders and classify it using the neural network.
function classify() {
  console.log("classify");
  const inputs = currentColor();
  neuralNetwork.classify([inputs.r, inputs.g, inputs.b], gotResults);
}

// Called when a classification is completed.
function gotResults(error, results) {
  console.log("got results");
  if (error) {
    statusP.innerText = "Error! See the console for details."
    console.error(error);
  } else {
    labelP.innerText = `label: ${results[0].label}, confidence: ${results[0].confidence.toFixed(2)}`;
    // Classify repeatedly during training only.
    if (!isTrained) {
      classify();
    }
  }
}
