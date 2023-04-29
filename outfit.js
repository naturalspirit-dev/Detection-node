const tf = require('@tensorflow/tfjs-node-gpu');
const bodyPix = require('@tensorflow-models/body-pix');
const fs = require('fs');
const sharp = require('sharp');

async function main(inputFilePath) {
  try {
    // Load the image file
    const imageBuffer = fs.readFileSync(inputFilePath);
    const {
      width,
      height
    } = await sharp(imageBuffer).metadata();
    const imageBufferRGB = await sharp(imageBuffer)
      .removeAlpha()
      .resize(width, height) // Maintain the original image dimensions
      .normalize() // Normalize the brightness and contrast of the image
      .toBuffer(); // Remove the alpha channel
    const imageTensor = tf.node.decodeImage(imageBufferRGB);

    // Load the BodyPix model with adjusted configuration
    const net = await bodyPix.load({
      architecture: 'MobileNetV1',
      outputStride: 16, // Use a higher output stride value for faster processing
      multiplier: 0.75, // Use the smallest available multiplier for faster processing
      quantBytes: 2,
    });
    console.log('Model loaded successfully');

    // Segment the person in the image
    const segmentation = await net.segmentMultiPersonParts(imageTensor, {
      scoreThreshold: 0.3, // Use a lower score threshold for more sensitive detection
    });
    console.log('Segmentation completed successfully:', segmentation);

    if (segmentation && segmentation.length) {
      if (segmentation[0] && segmentation[0].data) {
        // Code for processing the segmentation...
        // Convert the segmentation mask to a hair and body image with 50% transparency
        console.log('Creating mask image...');
        const maskRGBA = new Uint8ClampedArray(segmentation[0].data.length * 4);
        for (let i = 0; i < segmentation[0].data.length; i++) {
          if (i % 10000 === 0) {
            console.log(`Processing pixel ${i + 1}/${segmentation[0].data.length}`);
          }
          // Keep only the hair parts (label 1), paint them blue
          if (segmentation[0].data[i] === 1) {
            maskRGBA.set([0, 0, 0, 0], i * 4);
          }
          // Keep other body parts (labels >= 2), paint them gray
          else if (segmentation[0].data[i] >= 2) {
            maskRGBA.set([255, 255, 255, 255], i * 4);
          }
          // Make the background and other parts transparent
          else {
            maskRGBA.set([0, 0, 0, 0], i * 4);
          }
        }
        console.log('Mask image created.');
        const maskBufferRGBA = Buffer.from(maskRGBA.buffer);

        const maskImage = await sharp(maskBufferRGBA, {
          raw: {
            width: segmentation[0].width,
            height: segmentation[0].height,
            channels: 4,
          },
        }).png().toBuffer(); // Ensure mask image is in PNG format

        // Composite the original image with the segmentation mask
        const compositeImage = sharp(imageBufferRGB).png().composite([ // Ensure input image is in PNG format
          {
            input: maskImage,
            blend: 'dest-out'
          },
        ]).toFormat('png'); // Specify the output format

        // Save the composite image as a new file
        compositeImage.toFile(process.argv[3], (err, info) => {
          if (err) {
            console.error(err);
          } else {
            console.log('Output file saved:', info);
          }
        });
      } else {
        console.error('No person or body part detected in the image.');
      }
    } else {
      console.error('Segmentation result is not valid:', segmentation);
    }
  } catch (err) {
    console.error('An error occurred:', err);
  }
}

const inputFilePath = process.argv[2];

if (inputFilePath) {
  main(inputFilePath);
} else {
  console.error('Please provide an input file path as a command-line argument.');
}
