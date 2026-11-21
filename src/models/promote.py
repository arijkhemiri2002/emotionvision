"""Promote an MLflow run to `champion` in the Model Registry.

Usage:
    python -m src.models.promote --run-name efficientnet_b0_224
    python -m src.models.promote --run-id <run_id>

Then rebuild the deployment artifacts:
    dvc repro -f export
    dvc repro
"""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "emotionvision"


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-name")
    g.add_argument("--run-id")
    args = p.parse_args()

    client = MlflowClient()
    if args.run_name:
        exp = client.get_experiment_by_name("emotionvision")
        runs = client.search_runs([exp.experiment_id],
                                  filter_string=f"attributes.run_name = '{args.run_name}'",
                                  order_by=["attributes.start_time DESC"], max_results=1)
        if not runs:
            raise SystemExit(f"No run named '{args.run_name}'")
        run = runs[0]
    else:
        run = client.get_run(args.run_id)

    m = run.data.metrics
    print(f"[run] {run.info.run_name} ({run.info.run_id})  val_f1={m.get('best_val_macro_f1', 0):.4f} "
          f"test_acc={m.get('test_accuracy', 0):.4f} test_f1={m.get('test_macro_f1', 0):.4f}")

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    try:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    except Exception:  # registry without alias support -> tag fallback
        for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if v.tags.get("role") == "champion":
                client.delete_model_version_tag(MODEL_NAME, v.version, "role")
        client.set_model_version_tag(MODEL_NAME, mv.version, "role", "champion")
    print(f"[ok] {MODEL_NAME} v{mv.version} is now 'champion'")


if __name__ == "__main__":
    main()
