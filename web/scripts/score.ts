// Runs preprocessed photos through the model with onnxruntime-web, the runtime
// the site ships, so training/export.py measures what visitors actually get.
// (int8 kernels round slightly differently here than in native onnxruntime.)
//
// Usage: node scripts/score.ts <model.onnx> <inputs.bin> <probs.json>
//   inputs.bin: float32 little-endian, N x 3 x 224 x 224 (training/preprocess.py output)
//   probs.json: N arrays of Confidences, in Group order
import { readFileSync, writeFileSync } from "node:fs";
import * as ort from "onnxruntime-web/wasm";

const PER_PHOTO = 3 * 224 * 224;
const [modelPath, inputPath, outputPath] = process.argv.slice(2);
if (!modelPath || !inputPath || !outputPath) {
  console.error("usage: node scripts/score.ts <model.onnx> <inputs.bin> <probs.json>");
  process.exit(2);
}

ort.env.wasm.numThreads = 1; // multi-threaded wasm needs fetch, unavailable for local files in Node
const session = await ort.InferenceSession.create(readFileSync(modelPath));
const buf = readFileSync(inputPath);
const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const probs: number[][] = [];
for (let i = 0; i * PER_PHOTO < all.length; i++) {
  const input = all.slice(i * PER_PHOTO, (i + 1) * PER_PHOTO);
  const out = await session.run({ input: new ort.Tensor("float32", input, [1, 3, 224, 224]) });
  probs.push(Array.from(out.probs.data as Float32Array));
}
writeFileSync(outputPath, JSON.stringify(probs));
