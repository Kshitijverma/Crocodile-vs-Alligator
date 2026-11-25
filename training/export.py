"""Export the trained model for the browser and check the ship gate.

- ONNX (opset 17) with softmax built in: input "input" [N,3,224,224] -> "probs" [N,4].
- int8 dynamic quantisation of MatMul/Gemm weights only (nearly all of a ViT's
  weights; the patch-embedding Conv stays fp32 because ConvInteger has no CPU
  kernel). Static QDQ int8 was tried and cost ~7 points on this ViT.
- The int8 model is scored with onnxruntime-web (web/scripts/score.ts), the
  runtime the site ships: its int8 kernels round differently from native
  onnxruntime, enough to move accuracy by a point. fp32 is scored natively
  (the two runtimes agree exactly on fp32). int8 ships only if its test top-1
  is within 1 point of fp32.
- Threshold: the lowest of 0.50, 0.55, ... 0.95 at which named Verdicts on the
  val set are >= 94% right. See docs/adr/0001 for why 94% and not 92%.
- Ship gate (on the test set, at that threshold): named Verdicts >= 90% right
  AND at least half of photos named. Exits non-zero if it fails.
- Writes web/public/model/{model.onnx,labels.json,metrics.json},
  web/public/credits.json and the parity fixture under web/test/fixtures/.

Needs Node 24+ and `npm ci` in web/. Usage: python export.py
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from preprocess import load_eval, pixels_to_tensor
from train import ARCH, CHECKPOINTS, GROUPS, IMAGES, build_model

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parent / "web"
OUT = WEB / "public" / "model"
FIXTURES = WEB / "test" / "fixtures"
BUILD = ROOT / "checkpoints"
THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
VAL_NAMED_TARGET = 0.94
GATE_NAMED_ACCURACY = 0.90
GATE_NAMED_SHARE = 0.50
MAX_INT8_DROP = 0.01


def split_items(split: str) -> list[tuple[Path, int]]:
    return [
        (p, GROUPS.index(g))
        for g in GROUPS
        for p in sorted((IMAGES / split / g).glob("*.jpg"))
    ]


def native_probs(session: ort.InferenceSession, inputs: np.ndarray) -> np.ndarray:
    return np.concatenate([session.run(["probs"], {"input": x[None]})[0] for x in inputs])


def web_probs(model: Path, inputs: np.ndarray, name: str) -> np.ndarray:
    """Confidences from onnxruntime-web, via web/scripts/score.ts."""
    src, dst = BUILD / f"{name}.inputs.bin", BUILD / f"{name}.probs.json"
    inputs.astype("<f4").tofile(src)
    subprocess.run(["node", "scripts/score.ts", str(model), str(src), str(dst)], cwd=WEB, check=True)
    return np.array(json.loads(dst.read_text()), dtype=np.float64)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(1) == labels).mean())


def per_group_accuracy(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(1)
    return {g: float((pred[labels == i] == i).mean()) for i, g in enumerate(GROUPS)}


def named_stats(probs: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """(share of photos given a named Verdict, accuracy of those named Verdicts)."""
    named = probs.max(1) >= threshold
    if not named.any():
        return 0.0, 0.0
    return float(named.mean()), float((probs.argmax(1)[named] == labels[named]).mean())


def pick_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    for t in THRESHOLDS:
        if named_stats(probs, labels, t)[1] >= VAL_NAMED_TARGET:
            return t
    return THRESHOLDS[-1]


class WithSoftmax(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def main() -> int:
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINTS / "best.pt", map_location="cpu"))
    model.eval()

    fp32 = BUILD / "model.fp32.onnx"
    torch.onnx.export(
        WithSoftmax(model), torch.zeros(1, 3, 224, 224), fp32,
        input_names=["input"], output_names=["probs"],
        dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    prepped = BUILD / "model.prep.onnx"
    quant_pre_process(str(fp32), str(prepped))
    int8 = BUILD / "model.int8-dynamic.onnx"
    quantize_dynamic(str(prepped), str(int8), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])

    data = {}
    for split in ("val", "test"):
        items = split_items(split)
        data[split] = (np.stack([load_eval(p).numpy() for p, _ in items]), np.array([l for _, l in items]))
    (val_x, val_y), (test_x, test_y) = data["val"], data["test"]

    fp32_acc = accuracy(native_probs(ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]), test_x), test_y)
    print(f"fp32 (native) test top-1 {fp32_acc:.3f}  size {fp32.stat().st_size / 1e6:.1f} MB", flush=True)
    print("scoring int8 with onnxruntime-web (several minutes)...", flush=True)
    val_p, test_p = web_probs(int8, val_x, "val"), web_probs(int8, test_x, "test")
    int8_acc = accuracy(test_p, test_y)
    print(f"int8 (web)    test top-1 {int8_acc:.3f}  size {int8.stat().st_size / 1e6:.1f} MB", flush=True)
    if fp32_acc - int8_acc > MAX_INT8_DROP:
        print("EXPORT FAILED: int8 loses more than 1 point; fp32 is too big to ship.", file=sys.stderr)
        return 1

    threshold = pick_threshold(val_p, val_y)
    val_share, val_named = named_stats(val_p, val_y, threshold)
    share, named_acc = named_stats(test_p, test_y, threshold)
    metrics = {
        "model": f"{ARCH} fine-tuned (int8 dynamic)",
        "runtime": "onnxruntime-web (wasm)",
        "groups": GROUPS,
        "test_photos": len(test_y),
        "threshold": threshold,
        "test_named_share": round(share, 4),
        "test_named_accuracy": round(named_acc, 4),
        "val_named_share": round(val_share, 4),
        "val_named_accuracy": round(val_named, 4),
        "test_top1_accuracy": round(int8_acc, 4),
        "test_top1_accuracy_per_group": {k: round(v, 4) for k, v in per_group_accuracy(test_p, test_y).items()},
        "fp32_test_top1_accuracy": round(fp32_acc, 4),
        "threshold_rule": f"lowest threshold with val named accuracy >= {VAL_NAMED_TARGET}",
        "gate": f"test named accuracy >= {GATE_NAMED_ACCURACY} and named share >= {GATE_NAMED_SHARE}",
    }
    print(json.dumps(metrics, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(int8, OUT / "model.onnx")
    (OUT / "labels.json").write_text(json.dumps({"groups": GROUPS, "threshold": threshold}, indent=2) + "\n")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    with open(ROOT / "data" / "manifest.csv", encoding="utf-8") as fh:
        credits = [
            {"group": row["group"], "attribution": row["attribution"],
             "license": row["license"], "url": row["observation_url"]}
            for row in csv.DictReader(fh)
        ]
    (WEB / "public" / "credits.json").write_text(json.dumps(credits, indent=1) + "\n")

    # Parity fixture: a small PNG (lossless, so JS and Python decode identical
    # pixels), the tensor training/preprocess.py makes from it, and the
    # Confidences onnxruntime-web gives for that tensor.
    FIXTURES.mkdir(parents=True, exist_ok=True)
    with Image.open(split_items("test")[0][0]) as img:
        small = img.convert("RGB")
        small.thumbnail((320, 320))
        small.save(FIXTURES / "fixture.png")
    with Image.open(FIXTURES / "fixture.png") as img:
        x = pixels_to_tensor(np.asarray(img.convert("RGB"))).numpy()
    x.astype("<f4").tofile(FIXTURES / "input.bin")
    expected = web_probs(OUT / "model.onnx", x[None], "fixture")[0]
    (FIXTURES / "expected.json").write_text(json.dumps({"probs": [float(p) for p in expected]}, indent=2) + "\n")

    if named_acc < GATE_NAMED_ACCURACY or share < GATE_NAMED_SHARE:
        print(f"SHIP GATE FAILED: named accuracy {named_acc:.3f}, named share {share:.2f}", file=sys.stderr)
        return 1
    print("ship gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
