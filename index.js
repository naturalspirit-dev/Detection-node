const cv = require('opencv4nodejs');
const fs = require('fs');
const path = require('path');

const imagePath = 'Mine.png'; // Replace with your image path
const faceCascadePath = cv.HAAR_FRONTALFACE_DEFAULT;
const eyesCascadePath = cv.HAAR_EYE;
const lipsCascadePath = cv.HAAR_SMILE;

async function saveDetectedRegions() {
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

    let eyeCount = 0;
    let lipCount = 0;

    for (const face of faces.objects) {
      // Detect eyes within the face region
      const faceRegion = grayImage.getRegion(face);
      const eyes = await eyesCascade.detectMultiScaleAsync(faceRegion);

      for (const eye of eyes.objects) {
        // Adjust eye coordinates relative to the face
        const eyeRect = new cv.Rect(eye.x + face.x, eye.y + face.y, eye.width, eye.height);
      
        // Save eye region as a separate image
        const eyeRegion = image.getRegion(eyeRect);
        const eyeImagePath = path.join(__dirname, `eye_${++eyeCount}.png`);
        await cv.imwriteAsync(eyeImagePath, eyeRegion);
        console.log('Eye saved to', eyeImagePath);
      }

      // Detect lips within the face region
      const lips = await lipsCascade.detectMultiScaleAsync(faceRegion);

      for (const lip of lips.objects) {
        // Adjust lip coordinates relative to the face
        const lipRect = new cv.Rect(lip.x + face.x, lip.y + face.y, lip.width, lip.height);
      
        // Save lip region as a separate image
        const lipRegion = image.getRegion(lipRect);
        const lipImagePath = path.join(__dirname, `lip_${++lipCount}.png`);
        await cv.imwriteAsync(lipImagePath, lipRegion);
        console.log('Lip saved to', lipImagePath);
      }
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

saveDetectedRegions();
