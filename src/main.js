require("dotenv").config();// Initialize firebase admin
const fastify = require("fastify")({ logger: true });
const cors = require("@fastify/cors");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");

// Middlewares
fastify.register(require("@fastify/formbody"));
fastify.register(require("@fastify/multipart"), {
  preservePath: true,
  limits: {
    fileSize: 1024 * 1024 * 5, // 5MB
  },
});
fastify.register(cors, {
  origin: ["*"], // TODO : Change into domain or ip later on
});

// Image classification
fastify.post("/v1/image/predict", async (request, reply) => {
  const model = await tf.loadGraphModel(
    `file://src/model/converted/model.json`,
  );
  const file = await request.file();

  try {
    const imageBuffer = await file.toBuffer();
    const preprocessedImage = await sharp(imageBuffer)
      .resize(160, 160, {
        fit: sharp.fit.cover,
        position: sharp.strategy.entropy,
      }) // Resize the image to the expected input size
      .toFormat("jpeg") // Convert the image to JPEG
      .toBuffer(); // Get the raw buffer from the processed image

    // Decode the resized image Buffer into a tensor
    const imageTensor = tf.node.decodeImage(preprocessedImage, 3);
    // Normalize pixel values to [0, 1]
    const normalizedTensor = imageTensor.div(255.0);
    // Expand dimensions to match the expected input shape of [batchSize, height, width, channels]
    const expandedTensor = normalizedTensor.expandDims();

    const result = model.predict(expandedTensor);
    const resultArray = Array.from(await result.data());
    const classIndex = resultArray.indexOf(Math.max(...resultArray));
    const classes = ["sellable", "unsellable"];
    const classification = classes[classIndex];

    reply.send({
      message: "Image classification success",
      data: {
        result,
        classification,
      },
    });
  } catch (err) {
    reply.send(err);
  }

  reply.send({ message: "Oops, you're on the wrong path!" });
});

fastify.post("/v1/image/predict2", async (request, reply) => {
  const model = await tf.loadGraphModel(
    `file://src/model/converted/model.json`,
  );
  const file = await request.file();

  try {
    const imageBuffer = await file.toBuffer();
    const preprocessedImage = await sharp(imageBuffer)
      .resize(160, 160, {
        fit: sharp.fit.cover,
        position: sharp.strategy.entropy,
      }) // Resize the image to the expected input size
      .toFormat("jpeg") // Convert the image to JPEG
      .toBuffer(); // Get the raw buffer from the processed image

    // Decode the resized image Buffer into a tensor
    const imageTensor = tf.node.decodeImage(preprocessedImage, 3);
    // Normalize pixel values to [0, 1]
    const normalizedTensor = imageTensor.div(255.0);
    // Expand dimensions to match the expected input shape of [batchSize, height, width, channels]
    const expandedTensor = normalizedTensor.expandDims();

    const result = model.predict(expandedTensor);
    const resultArray = Array.from(await result.data());
    const classIndex = resultArray.indexOf(Math.max(...resultArray));
    const classes = ["sellable", "unsellable"];
    const classification = classes[classIndex];

    reply.send({
      message: "Image classification success",
      data: {
        result,
        classification,
      },
    });
  } catch (err) {
    reply.send(err);
  }

  reply.send({ message: "Oops, you're on the wrong path!" });
});

fastify.get("/", async (request, reply) => {
  reply.send({ message: "Hello World" });
});
// 1 endpoint

// Run the server!
const PORT = process.env.PORT || 8000;
const HOST = process.env.HOST || "0.0.0.0";

const start = async () => {
  try {
    await fastify.listen({ host: HOST, port: PORT });
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};
start();
