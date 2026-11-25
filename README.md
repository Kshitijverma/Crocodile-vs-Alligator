# Crocodilian Classifier

Upload a photo and find out whether it shows an **alligator**, **caiman**, **crocodile** or **gharial**. If the model isn't confident, it says **Can't tell** instead of guessing.

**Live:** https://crocodile-vs-alligator.web.app

Everything runs in your browser: a fine-tuned DINOv2 ViT-S/14 model runs through [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), so your photo is never uploaded anywhere. See [`docs/adr/0001`](docs/adr/0001-in-browser-inference-retrained-model.md) for why there's no backend.

## What counts as what

The four Groups are defined by taxonomy. See [`CONTEXT.md`](CONTEXT.md) for the full glossary.

| Group | Genera |
|---|---|
| Alligator | *Alligator* |
| Caiman | *Caiman*, *Melanosuchus*, *Paleosuchus* |
| Crocodile | *Crocodylus*, *Mecistops*, *Osteolaemus* |
| Gharial | *Gavialis*, *Tomistoma* |

How often named answers are right, how many photos get one, and the Can't-tell Threshold are all in [`web/public/model/metrics.json`](web/public/model/metrics.json). The model only names a Group when it is confident; distant, night or murky photos usually get Can't tell.

## Layout

- `web/` is the site (Vite + React 18 + TypeScript).
  - `npm run dev` starts it locally.
  - `npm test` runs the verdict rule tests and the preprocessing parity test.
  - `npm run build` builds it to `web/dist`.
- `training/` holds the reproducible training pipeline (Python, CPU is enough).
- `.github/workflows/ci.yml` lints, tests and builds on every PR, and deploys to Firebase Hosting on every push to `master`.

## Retraining

```bash
cd training
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_inat.py      # ~2,400 photos from iNaturalist, into data/images/
.venv/bin/python train.py           # fine-tune (~35 min on a 4-core CPU) -> checkpoints/best.pt
.venv/bin/python export.py          # ONNX + int8, metrics, credits, parity fixture -> web/
```

`export.py` needs Node 24+ and `npm ci` in `web/`: it scores the int8 model with onnxruntime-web (`web/scripts/score.ts`), the same runtime visitors use. It exits with an error unless, on the test set, named answers are at least 90% right and at least half of photos get one. `training/preprocess.py` and `web/src/preprocess.ts` must stay identical. `web/test/parity.test.ts` checks this.

## Credits

Training photos come from research-grade [iNaturalist](https://www.inaturalist.org) observations, used under CC0, CC BY and CC BY-NC licences. Every photo is listed with its observer and licence in [`training/data/manifest.csv`](training/data/manifest.csv) and on the site's Credits page.

This started in 2019 as a fastai crocodile-vs-alligator demo. It was rebuilt in 2026 with two more Groups and no server.
