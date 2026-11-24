
import argparse

import joblib
import pandas as pd

import config as cfg


def load_artefact():
    if not cfg.MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    return joblib.load(cfg.MODEL_PATH)


def validate_inputs(values: dict) -> list:
    
    errors = []
    for feat, (lo, hi, *_rest, label) in cfg.FEATURE_INFO.items():
        v = values.get(feat)
        if v is None:
            errors.append(f"{label} is missing.")
        elif not (lo <= float(v) <= hi):
            errors.append(f"{label} must be between {lo} and {hi} (got {v}).")
    return errors


def recommend(values: dict, top_k: int = 3, artefact=None):
    """Return list of (crop, probability) sorted high -> low."""
    errors = validate_inputs(values)
    if errors:
        raise ValueError("; ".join(errors))
    artefact = artefact or load_artefact()
    row = pd.DataFrame([values])[artefact["features"]]
    proba = artefact["pipeline"].predict_proba(row)[0]
    ranked = sorted(zip(artefact["classes"], proba), key=lambda t: t[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crop recommendation from the command line")
    for f, (_, _, default, _, label) in cfg.FEATURE_INFO.items():
        ap.add_argument(f"--{f}", type=float, default=default, help=label)
    args = vars(ap.parse_args())
    for crop, p in recommend(args):
        print(f"{crop:12s} {p:.1%}")
