"""Download a balanced crocodilian photo dataset from iNaturalist.

Pulls research-grade (community-verified, wild) observations for each Group,
keeps one openly licensed photo per Observation, splits 80/10/10 by
Observation, and writes images under training/data/images/ plus
training/data/manifest.csv (committed, so Credits can be rebuilt).

Usage: python fetch_inat.py [--per-group 600]
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import requests

# Group -> iNaturalist genus taxon IDs (see CONTEXT.md for the definitions).
GROUPS: dict[str, dict[str, int]] = {
    "alligator": {"Alligator": 26154},
    "caiman": {"Caiman": 116638, "Melanosuchus": 116646, "Paleosuchus": 116648},
    "crocodile": {"Crocodylus": 26042, "Mecistops": 194089, "Osteolaemus": 26135},
    "gharial": {"Gavialis": 116641, "Tomistoma": 116651},
}
LICENSES = ("cc0", "cc-by", "cc-by-nc")
API = "https://api.inaturalist.org/v1/observations"
HEADERS = {"User-Agent": "crocodilian-classifier-training/1.0"}
SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))
SEED = 1234

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFEST_FIELDS = [
    "group", "split", "file", "observation_id", "photo_id",
    "license", "attribution", "observation_url",
]


def fetch_observations(taxon_ids: list[int], limit: int) -> list[dict]:
    """Newest-first research-grade observations with an allowed photo licence."""
    rows: list[dict] = []
    id_below: int | None = None
    while len(rows) < limit:
        params = {
            "taxon_id": ",".join(map(str, taxon_ids)),
            "quality_grade": "research",
            "photo_license": ",".join(LICENSES),
            "per_page": 200,
            "order_by": "id",
            "order": "desc",
        }
        if id_below:
            params["id_below"] = id_below
        resp = requests.get(API, params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        results = resp.json()["results"]
        if not results:
            break
        for obs in results:
            photo = next(
                (p for p in obs.get("photos", []) if p.get("license_code") in LICENSES),
                None,
            )
            if photo:
                rows.append({"obs": obs, "photo": photo})
        id_below = results[-1]["id"]
        time.sleep(1.0)  # iNaturalist asks for <= 60 requests/minute
    return rows[:limit]


def assign_splits(n: int, rng: random.Random) -> list[str]:
    order = list(range(n))
    rng.shuffle(order)
    labels = [""] * n
    start = 0
    for i, (name, frac) in enumerate(SPLITS):
        end = n if i == len(SPLITS) - 1 else start + round(n * frac)
        for idx in order[start:end]:
            labels[idx] = name
        start = end
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=600)
    args = parser.parse_args()

    rng = random.Random(SEED)
    manifest: list[dict] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for group, genera in GROUPS.items():
        rows = fetch_observations(list(genera.values()), args.per_group)
        print(f"{group}: {len(rows)} observations")
        for row, split in zip(rows, assign_splits(len(rows), rng)):
            obs, photo = row["obs"], row["photo"]
            rel = Path("images") / split / group / f"{photo['id']}.jpg"
            dest = DATA / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                url = photo["url"].replace("/square.", "/medium.")
                img = session.get(url, timeout=60)
                img.raise_for_status()
                dest.write_bytes(img.content)
                time.sleep(0.3)
            manifest.append({
                "group": group,
                "split": split,
                "file": rel.as_posix(),
                "observation_id": obs["id"],
                "photo_id": photo["id"],
                "license": photo["license_code"],
                "attribution": photo.get("attribution", ""),
                "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
            })

    with open(DATA / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"wrote {len(manifest)} rows to {DATA / 'manifest.csv'}")


if __name__ == "__main__":
    main()
