import { loadImage, ImageData } from 'canvas';

export const asyncLoadImage = async (src: string) => {
  return loadImage(src);
}

export const getRobin = async () => {
  return asyncLoadImage("https://cdn.jsdelivr.net/gh/ml5js/ml5-library@main/assets/bird.jpg");
}

export const randomImageData = (width = 200, height = 100) => {
  const length = width * height * 4; // 4 channels - RGBA
  // Create an array of random pixel values
  const array = Uint8ClampedArray.from(
    { length },
    () => Math.floor(Math.random() * 256)
  );
  // Initialize a new ImageData object
  return new ImageData(array, width, height);
}
