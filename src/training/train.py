"""Train a facial expression classifier and log everything to MLflow.

Examples:
    python -m src.training.train --model baseline_cnn --no-pretrained
    python -m src.training.train --model efficientnet_b0
    python -m src.training.train --model resnet50 --register

MLflow: set MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
(DagsHub) before running; otherwise runs are stored locally in ./mlruns.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from torch import nn

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_loaders, class_weights
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, predict, report
from src.models.factory import build_model, set_backbone_trainable

REGISTERED_MODEL_NAME = "emotionvision"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(defaults: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=defaults["model"])
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=defaults["pretrained"])
    p.add_argument("--input-size", type=int, default=defaults["input_size"])
    p.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument("--lr", type=float, default=defaults["lr"])
    p.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    p.add_argument("--freeze-backbone-epochs", type=int, default=defaults["freeze_backbone_epochs"])
    p.add_argument("--backbone-lr-mult", type=float, default=defaults["backbone_lr_mult"])
    p.add_argument("--label-smoothing", type=float, default=defaults["label_smoothing"])
    p.add_argument("--class-weighting", default=defaults["class_weighting"],
                   choices=["none", "inv", "sqrt_inv"])
    p.add_argument("--patience", type=int, default=defaults["patience"])
    p.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    p.add_argument("--max-batches", type=int, default=0, help="debug: limit batches per epoch")
    p.add_argument("--register", action="store_true", help="register the model in the MLflow registry")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def make_optimizer(model: nn.Module, lr: float, wd: float, backbone_mult: float):
    head = model.get_classifier() if hasattr(model, "get_classifier") else model.classifier
    head_ids = {id(p) for p in head.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids and p.requires_grad]
    body_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    groups = [{"params": head_params, "lr": lr}]
    if body_params:
        groups.append({"params": body_params, "lr": lr * backbone_mult})
    return torch.optim.AdamW(groups, weight_decay=wd)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_batches=0):
    model.train()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    args = parse_args(params["train"])
    data_cfg = params["data"]
    classes = data_cfg["classes"]
    seed_everything(params["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    loaders, dsets = build_loaders(data_cfg["processed_dir"], classes, args.input_size,
                                   args.batch_size, args.num_workers)
    if args.max_batches:  # debug mode: also shrink eval sets
        for split in ("val", "test"):
            sub = torch.utils.data.Subset(dsets[split], range(min(len(dsets[split]), args.max_batches * args.batch_size)))
            loaders[split] = torch.utils.data.DataLoader(sub, batch_size=args.batch_size)

    weights = class_weights(dsets["train"].targets, len(classes), args.class_weighting)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=args.label_smoothing,
    )
    eval_criterion = nn.CrossEntropyLoss()

    model = build_model(args.model, len(classes), args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    freeze_epochs = args.freeze_backbone_epochs if args.pretrained else 0
    if freeze_epochs:
        set_backbone_trainable(model, False)
    optimizer = make_optimizer(model, args.lr, args.weight_decay, args.backbone_lr_mult)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(freeze_epochs, 1))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{args.model}_best.pt"

    mlflow.set_experiment("emotionvision")
    with mlflow.start_run(run_name=args.run_name or args.model):
        mlflow.log_params({**vars(args), "n_params": n_params, "device": device.type,
                           "classes": ",".join(classes), "train_size": len(dsets["train"])})
        if weights is not None:
            mlflow.log_params({f"w_{c}": round(float(w), 3) for c, w in zip(classes, weights)})

        best_f1, bad_epochs = -1.0, 0
        for epoch in range(1, args.epochs + 1):
            if freeze_epochs and epoch == freeze_epochs + 1:
                print(f"[epoch {epoch}] unfreezing backbone -> full fine-tuning")
                set_backbone_trainable(model, True)
                optimizer = make_optimizer(model, args.lr, args.weight_decay, args.backbone_lr_mult)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - freeze_epochs)
            elif epoch == 1 and not freeze_epochs:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            t0 = time.time()
            train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler,
                                         device, args.max_batches)
            scheduler.step()
            y_true, y_pred, val_loss = predict(model, loaders["val"], device, eval_criterion)
            m = compute_metrics(y_true, y_pred)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
                                "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
            print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_acc={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f} ({time.time() - t0:.0f}s)")

            if m["macro_f1"] > best_f1:
                best_f1, bad_epochs = m["macro_f1"], 0
                torch.save(model.state_dict(), best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"[early stop] no improvement for {args.patience} epochs")
                    break

        # ---- Final evaluation on the test set with the best checkpoint ----
        model.load_state_dict(torch.load(best_path, map_location=device))
        y_true, y_pred, test_loss = predict(model, loaders["test"], device, eval_criterion)
        tm = compute_metrics(y_true, y_pred)
        mlflow.log_metrics({"best_val_macro_f1": best_f1, "test_loss": test_loss,
                            **{f"test_{k}": v for k, v in tm.items()}})
        rep = report(y_true, y_pred, classes)
        print(rep)

        cm_path = fig_dir / f"confusion_matrix_{args.model}.png"
        plot_confusion_matrix(y_true, y_pred, classes, str(cm_path), f"{args.model} - test set")
        rep_path = Path("reports") / f"classification_report_{args.model}.txt"
        rep_path.write_text(rep)

        # Config needed later by ONNX export / API / Streamlit
        cfg = {"model": args.model, "classes": classes, "input_size": args.input_size,
               "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "grayscale_to_rgb": True}
        cfg_path = out_dir / "model_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        for p in (cm_path, rep_path, cfg_path, best_path, "params.yaml"):
            mlflow.log_artifact(str(p))

        model.cpu().eval()
        example = np.random.rand(1, 3, args.input_size, args.input_size).astype(np.float32)
        # MLflow 3 renamed `artifact_path` to `name`
        if int(mlflow.__version__.split(".")[0]) >= 3:
            name_kw = {"name": "model", "serialization_format": "pickle"}
        else:
            name_kw = {"artifact_path": "model"}
        mlflow.pytorch.log_model(
            model, input_example=example,
            registered_model_name=REGISTERED_MODEL_NAME if args.register else None,
            **name_kw,
        )
        print(f"[done] test_acc={tm['accuracy']:.4f} test_macro_f1={tm['macro_f1']:.4f}")
        if os.getenv("MLFLOW_TRACKING_URI"):
            print(f"[mlflow] run logged to {os.getenv('MLFLOW_TRACKING_URI')}")


if __name__ == "__main__":
    main()
