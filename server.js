const tf = require('@tensorflow/tfjs-node-gpu');
const bodyPix = require('@tensorflow-models/body-pix');
const fs = require('fs');
const sharp = require('sharp');

async function main(inputFilePath) {
  try {
    // Load the image file
    const imageBuffer = fs.readFileSync(inputFilePath);
    const imageBufferRGBA = await sharp(imageBuffer)
      .ensureAlpha() // Ensure the image has an alpha channel
      .resize(128, 128) // Resize the image to a fixed size, e.g., 512x512 pixels
      .normalize() // Normalize the brightness and contrast of the image
      .toBuffer();

    // Convert RGBA to RGB
    const imageBufferRGB = await sharp(imageBufferRGBA)
      .removeAlpha()
      .toBuffer();
    const imageTensor = tf.node.decodeImage(imageBufferRGB, 3); // Pass the number of channels (3 for RGB)

    // Load the BodyPix model with adjusted configuration
    const net = await bodyPix.load({
      architecture: 'MobileNetV1',
      outputStride: 8, // Use a lower output stride value for better accuracy
      multiplier: 0.75,
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
        const maskRGBA = segmentation[0].data.reduce((acc, x, index) => {
          if (index % 10000 === 0) {
            console.log(`Processing pixel ${index + 1}/${segmentation[0].data.length}`);
          }
          // Keep only the hair parts (label 1), paint them blue
          if (x === 1) {
            return acc.concat([0, 0, 0, 0]);
          }
          // Keep other body parts (labels >= 2), paint them gray
          else if (x >= 2) {
            return acc.concat([-1,-1,-1,-1]);
          }
          // Make the background and other parts transparent
          else {
            return acc.concat([0, 0, 0, 0]);
          }
        }, []);
        console.log('Mask image created.');
        const maskBufferRGBA = Buffer.from(maskRGBA);

        // Create a new buffer containing only the alpha channel from the input image
        const inputAlphaChannel = await sharp(imageBufferRGBA)
          .extractChannel(3)
          .raw()
          .toBuffer();
        const inputAlphaChannelBuffer = Buffer.from(inputAlphaChannel);

        // Combine the input alpha channel with the segmentation mask
        const combinedMask = await sharp(maskBufferRGBA, {
            raw: {
              width: segmentation[0].width,
              height: segmentation[0].height,
              channels: 4,
            },
          })
          .composite([{
            input: inputAlphaChannelBuffer,
            raw: {
              width: segmentation[0].width,
              height: segmentation[0].height,
              channels: 1
            },
            blend: 'dest-in'
          }])
          .toBuffer();

        const combinedMaskImage = await sharp(combinedMask, {
          raw: {
            width: segmentation[0].width,
            height: segmentation[0].height,
            channels: 4,
          },
        }).png().toBuffer(); // Ensure mask image is in PNG format

        // Composite the original image with the combined mask
        const compositeImage = sharp(imageBufferRGBA).png().composite([ // Ensure input image is in PNG format
          {
            input: combinedMaskImage,
            blend: 'over'
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