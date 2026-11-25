import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web/wasm";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import { INPUT_SIZE, preprocess } from "../src/preprocess";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

// training/export.py writes these fixtures:
// - fixture.png: a test-set photo, as lossless PNG so both sides decode identical pixels
// - input.bin:   the float32 tensor training/preprocess.py makes from it
// - expected.json: the Confidences onnxruntime-web gave for that tensor at export time
function fixture() {
  const png = PNG.sync.read(readFileSync(here("./fixtures/fixture.png")));
  const buf = readFileSync(here("./fixtures/input.bin"));
  const pythonInput = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const expected: number[] = JSON.parse(readFileSync(here("./fixtures/expected.json"), "utf8")).probs;
  return { png, pythonInput, expected };
}

const argmax = (xs: ArrayLike<number>) => Array.from(xs).indexOf(Math.max(...Array.from(xs)));

describe("preprocessing parity with training/preprocess.py", () => {
  it("turns the fixture photo into the same tensor as Python", () => {
    const { png, pythonInput } = fixture();
    const input = preprocess({ data: png.data, width: png.width, height: png.height });

    expect(input).toHaveLength(pythonInput.length);
    let worst = 0;
    for (let i = 0; i < input.length; i++) worst = Math.max(worst, Math.abs(input[i] - pythonInput[i]));
    expect(worst).toBeLessThan(1e-4);
  });
});

// Catches a model the browser runtime can't execute, or an onnxruntime-web
// upgrade that changes the answers. Loose tolerance: int8 kernel rounding may
// shift between versions without changing the Verdict.
describe("model in onnxruntime-web", () => {
  it("gives the fixture photo the Confidences recorded at export", async () => {
    const { pythonInput, expected } = fixture();
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(readFileSync(here("../public/model/model.onnx")));
    const out = await session.run({
      input: new ort.Tensor("float32", pythonInput, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    });
    const probs = Array.from(out.probs.data as Float32Array);

    expect(probs).toHaveLength(expected.length);
    expect(argmax(probs)).toBe(argmax(expected));
    probs.forEach((p, i) => expect(Math.abs(p - expected[i])).toBeLessThan(2e-2));
  }, 60_000);
});
