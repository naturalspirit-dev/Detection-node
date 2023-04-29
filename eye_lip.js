const cv = require('opencv4nodejs');
const fs = require('fs');
const path = require('path');

const imagePath = '1.png'; // Replace with your image path
const faceCascadePath = cv.HAAR_FRONTALFACE_DEFAULT;// 'haarcascade_frontalface_default.xml';
const eyesCascadePath = cv.HAAR_EYE; // 'haarcascade_eye.xml';
const lipsCascadePath = cv.HAAR_SMILE; // 'haarcascade_smile.xml';

async function detectEyesAndLips() {
  try {
    // Load the image
    const image = await cv.imreadAsync(imagePath);

    // Load the Haar cascades
    const faceCascade = new cv.CascadeClassifier(faceCascadePath);
    const eyesCascade = new cv.CascadeClassifier(eyesCascadePath);
    const lipsCascade = new cv.CascadeClassifier(lipsCascadePath);

    // Convert the image to grayscale
    const grayImage = image.bgrToGray();

    // Detect faces
    const faces = await faceCascade.detectMultiScaleAsync(grayImage);

    for (const face of faces.objects) {
      // Draw face bounding box
      image.drawRectangle(face, new cv.Vec(0, 255, 0), 2);

      // Detect eyes within the face region
      const faceRegion = grayImage.getRegion(face);
      const eyes = await eyesCascade.detectMultiScaleAsync(faceRegion);

      for (const eye of eyes.objects) {
        // Adjust eye coordinates relative to the face
        const eyeRect = new cv.Rect(eye.x + face.x, eye.y + face.y, eye.width, eye.height);
      
        // Draw eye bounding box
        image.drawRectangle(eyeRect, new cv.Vec(255, 0, 0), 2);
      }

      // Detect lips within the face region
      const lips = await lipsCascade.detectMultiScaleAsync(faceRegion);

      for (const lip of lips.objects) {
        // Adjust lip coordinates relative to the face
        const lipRect = new cv.Rect(lip.x + face.x, lip.y + face.y, lip.width, lip.height);
      
        // Draw lip bounding box
        image.drawRectangle(lipRect, new cv.Vec(0, 0, 255), 2);
      }
    }

    // Save the output image
    const outputImagePath = path.join(__dirname, 'output.jpg');
    await cv.imwriteAsync(outputImagePath, image);
    console.log('Output saved to', outputImagePath);
  } catch (error) {
    console.error('Error:', error);
  }
}

detectEyesAndLips();