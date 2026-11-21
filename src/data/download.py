"""Download FER2013 (Kaggle) + FER+ labels (Microsoft) into data/raw/.

Output:
    data/raw/fer2013.csv      -> emotion, Usage, pixels (35,887 rows, 48x48 grayscale)
    data/raw/fer2013new.csv   -> FER+ votes from 10 annotators, same row order

Prerequisites:
    1. pip install kaggle
    2. Kaggle API token in %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
       or ~/.kaggle/kaggle.json (Linux/Mac)
    3. Accept the competition rules once on the website:
       https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/rules

Usage:
    python -m src.data.download
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
COMPETITION = "challenges-in-representation-learning-facial-expression-recognition-challenge"
FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
FALLBACK_DATASET = "deadskull7/fer2013"  # public mirror of the same fer2013.csv
EXPECTED_ROWS = 35_887


def download_ferplus_labels(dest: Path) -> Path:
    out = dest / "fer2013new.csv"
    if out.exists():
        print(f"[skip] {out} already exists")
        return out
    print(f"[download] FER+ labels -> {out}")
    urllib.request.urlretrieve(FERPLUS_URL, out)
    return out


def download_fer2013(dest: Path) -> Path:
    out = dest / "fer2013.csv"
    if out.exists():
        print(f"[skip] {out} already exists")
        return out

    from kaggle.api.kaggle_api_extended import KaggleApi  # imported here: needs kaggle.json

    tmp = dest / "_kaggle_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    try:
        print("[download] FER2013 from Kaggle (official competition)...")
        api.competition_download_file(COMPETITION, "fer2013.tar.gz", path=str(tmp), quiet=False)
    except Exception as e:  # 403 if competition rules were not accepted
        print(f"[warn] Competition download failed ({e}).")
        print(f"[download] Falling back to Kaggle dataset '{FALLBACK_DATASET}'...")
        api.dataset_download_files(FALLBACK_DATASET, path=str(tmp), quiet=False, unzip=True)

    # Extract every archive we got (zip, then the inner tar.gz if present)
    for z in tmp.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
    for t in tmp.rglob("*.tar.gz"):
        with tarfile.open(t) as tf:
            tf.extractall(tmp)

    # The official CSV is either fer2013.csv or icml_face_data.csv (same content/order)
    candidates = list(tmp.rglob("fer2013.csv")) + list(tmp.rglob("icml_face_data.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No fer2013.csv / icml_face_data.csv found in {tmp}. "
            "Did you accept the competition rules on Kaggle?"
        )

    df = pd.read_csv(candidates[0])
    df.columns = [c.strip() for c in df.columns]  # icml_face_data.csv has ' Usage', ' pixels'
    df = df[["emotion", "Usage", "pixels"]]
    df.to_csv(out, index=False)

    shutil.rmtree(tmp)
    return out


def validate(fer_path: Path, ferplus_path: Path) -> None:
    fer = pd.read_csv(fer_path)
    ferplus = pd.read_csv(ferplus_path)

    assert len(fer) == EXPECTED_ROWS, f"fer2013.csv: {len(fer)} rows, expected {EXPECTED_ROWS}"
    assert len(ferplus) == EXPECTED_ROWS, f"fer2013new.csv: {len(ferplus)} rows, expected {EXPECTED_ROWS}"
    assert (fer["Usage"].values == ferplus["Usage"].values).all(), "Row order mismatch FER vs FER+"

    print("\n[ok] Both files present, 35,887 rows, row order aligned.")
    print(fer["Usage"].value_counts().to_string())


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ferplus = download_ferplus_labels(RAW_DIR)
    fer = download_fer2013(RAW_DIR)
    validate(fer, ferplus)


if __name__ == "__main__":
    main()
