"""DVC stage `prepare`: FER2013 pixels + FER+ votes -> PNG folders (7 classes).

Label rule (FER+ "majority" mode, Barsoum et al. 2016):
    1. emotions with a single vote are treated as noise and set to 0
    2. the image keeps the class that has > 50% of the remaining votes
    3. images whose winner is contempt / unknown / NF (not a face), or with
       no clear majority, are dropped

Splits: Training -> train, PublicTest -> val, PrivateTest -> test

Output:
    data/processed/{train,val,test}/{class}/ferXXXXXXX.png
    reports/data_stats.json            (DVC metrics)
    reports/figures/class_distribution.png

Usage:
    python -m src.data.prepare
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

FERPLUS_COLS = [
    "neutral", "happiness", "surprise", "sadness", "anger",
    "disgust", "fear", "contempt", "unknown", "NF",
]
SPLITS = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


def majority_label(votes: np.ndarray, keep_idx: list[int]) -> int | None:
    """Return the FER+ column index of the majority class, or None to drop."""
    v = votes.astype(float).copy()
    v[v <= 1] = 0                     # remove single-vote outliers
    total = v.sum()
    if total == 0:
        return None
    winner = int(v.argmax())
    if v[winner] <= 0.5 * total:      # no clear majority
        return None
    if winner not in keep_idx:        # contempt / unknown / NF
        return None
    return winner


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["data"]
    raw = Path(params["raw_dir"])
    out = Path(params["processed_dir"])
    classes: list[str] = params["classes"]
    keep_idx = [FERPLUS_COLS.index(c) for c in classes]

    fer = pd.read_csv(raw / "fer2013.csv")
    ferplus = pd.read_csv(raw / "fer2013new.csv")
    assert len(fer) == len(ferplus), "FER and FER+ must have the same number of rows"

    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS.values():
        for c in classes:
            (out / split / c).mkdir(parents=True, exist_ok=True)

    counts = {s: {c: 0 for c in classes} for s in SPLITS.values()}
    dropped = 0
    votes_all = ferplus[FERPLUS_COLS].to_numpy()

    for i in range(len(fer)):
        name = ferplus.at[i, "Image name"]
        if not isinstance(name, str):          # image removed by FER+ authors
            dropped += 1
            continue
        label = majority_label(votes_all[i], keep_idx)
        if label is None:
            dropped += 1
            continue

        split = SPLITS[fer.at[i, "Usage"]]
        cls = FERPLUS_COLS[label]
        pixels = np.array(fer.at[i, "pixels"].split(), dtype=np.uint8).reshape(48, 48)
        Image.fromarray(pixels).save(out / split / cls / name)
        counts[split][cls] += 1

    stats = {
        "total_kept": sum(sum(d.values()) for d in counts.values()),
        "dropped": dropped,
        **{f"{s}_size": sum(d.values()) for s, d in counts.items()},
        "per_class": counts,
    }
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    json.dump(stats, open("reports/data_stats.json", "w"), indent=2)

    # Class distribution plot
    df = pd.DataFrame(counts).loc[classes]
    ax = df.plot(kind="bar", figsize=(9, 4.5), width=0.8)
    ax.set_title("FER+ class distribution (majority vote, 7 classes)")
    ax.set_ylabel("Images")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/figures/class_distribution.png", dpi=150)

    print(json.dumps({k: v for k, v in stats.items() if k != "per_class"}, indent=2))
    print(df.to_string())


if __name__ == "__main__":
    main()
