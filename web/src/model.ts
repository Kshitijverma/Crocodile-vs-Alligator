import * as ort from "onnxruntime-web/wasm";
import { INPUT_SIZE, preprocess } from "./preprocess";
import { verdict, type Verdict } from "./verdict";

/** Longest side photos are scaled to before preprocessing, roughly the size of the training photos. */
const MAX_SIDE = 512;

interface Labels {
  groups: string[];
  threshold: number;
}

export interface Classification {
  verdict: Verdict;
  confidences: { group: string; confidence: number }[];
}

let loading: Promise<{ session: ort.InferenceSession; labels: Labels }> | null = null;

/** Load the model once, on first use. */
function load() {
  loading ??= (async () => {
    const [session, labels] = await Promise.all([
      ort.InferenceSession.create("/model/model.onnx", { executionProviders: ["wasm"] }),
      fetch("/model/labels.json").then((r) => r.json() as Promise<Labels>),
    ]);
    return { session, labels };
  })();
  return loading;
}

/** Draw the photo at most MAX_SIDE px on its longest side and read back RGBA pixels. */
async function readPixels(file: File) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D is not available in this browser.");
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return ctx.getImageData(0, 0, width, height);
}

export async function classify(file: File): Promise<Classification> {
  const [{ session, labels }, pixels] = await Promise.all([load(), readPixels(file)]);
  const input = new ort.Tensor("float32", preprocess(pixels), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const out = await session.run({ input });
  const probs = out.probs.data as Float32Array;
  return {
    verdict: verdict(probs, labels.groups, labels.threshold),
    confidences: labels.groups.map((group, i) => ({ group, confidence: probs[i] })),
  };
}
