let img;
let poseNet;
let poses = [];

function setup() {
  createCanvas(640, 360);

  // create an image using the p5 dom library
  // call modelReady() when it is loaded
  img = createImg("data/runner.jpg", imageReady);
  // set the image size to the size of the canvas
  img.size(width, height);

  img.hide(); // hide the image in the browser
  frameRate(1); // set the frameRate to 1 since we don't need it to be running quickly in this case
}

// when the image is ready, then load up poseNet
function imageReady() {
  // set some options
  const options = {
    minConfidence: 0.1,
    inputResolution: { width, height },
  };

  // assign poseNet
  poseNet = ml5.poseNet(modelReady, options);
  // This sets up an event that listens to 'pose' events
  poseNet.on("pose", (results) => {
    poses = results;
  });
}

// when poseNet is ready, do the detection
function modelReady() {
  select("#status").html("Model Loaded");

  // When the model is ready, run the singlePose() function...
  // If/When a pose is detected, poseNet.on('pose', ...) will be listening for the detection results
  // in the draw() loop, if there are any poses, then carry out the draw commands
  poseNet.singlePose(img);
}

// draw() will not show anything until poses are found
function draw() {
  if (poses.length > 0) {
    image(img, 0, 0, width, height);
    drawSkeleton(poses);
    drawKeypoints(poses);
    noLoop(); // stop looping when the poses are estimated
  }
}

// The following comes from https://ml5js.org/docs/posenet-webcam
// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  poses.forEach(pose => {
    // For each pose detected, loop through all the keypoints
    // A keypoint is an object describing a body part (like rightArm or leftShoulder)
    pose.pose.keypoints.forEach(keypoint => {
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255);
        stroke(20);
        strokeWeight(4);
        ellipse(round(keypoint.position.x), round(keypoint.position.y), 8, 8);
      }
    });
  });
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  poses.forEach(pose => {
    const {skeleton} = pose;
    // For every skeleton, loop through all body connections
    skeleton.forEach(bodyPart => {
      const [partA, partB] = bodyPart;
      stroke(255);
      strokeWeight(1);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    });
  });
}
