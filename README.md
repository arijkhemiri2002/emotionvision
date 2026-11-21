# EmotionVision MLOps

Facial expression recognition (7 classes) with transfer learning, Grad-CAM explainability,
ONNX optimization and a full MLOps pipeline (DVC, MLflow, FastAPI, Streamlit, Docker, GitHub Actions).

**Classes:** neutral, happiness, surprise, sadness, anger, disgust, fear
**Data:** FER2013 images + FER+ labels (10 annotators per image, majority vote)

## Project structure

```
emotionvision/
├── data/
│   ├── raw/            # fer2013.csv + fer2013new.csv (DVC)
│   └── processed/      # train/val/test PNG folders (DVC)
├── src/
│   ├── data/           # download, prepare
│   ├── models/         # baseline CNN, EfficientNet, ResNet
│   ├── training/       # training loop + MLflow logging
│   ├── evaluation/     # metrics, confusion matrix
│   └── explain/        # Grad-CAM
├── api/                # FastAPI (async, Pydantic, ONNX Runtime)
├── app/                # Streamlit frontend
├── tests/              # Pytest
├── notebooks/          # Kaggle training notebook, EDA
├── models/             # exported models (DVC)
├── reports/figures/
├── params.yaml
└── dvc.yaml
```

## Setup

```bash
pip install -r requirements.txt
python -m src.data.download      # needs a Kaggle API token
dvc pull                         # or pull data from DagsHub
```

> Datasets are for non-commercial research/education only and are **not redistributed** in this repository.
