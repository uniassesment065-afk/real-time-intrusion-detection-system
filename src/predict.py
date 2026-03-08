import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

DEFAULT_MODEL_PATH = "models/ids_model.pkl"
DEFAULT_PIPELINE_PATH = "models/pipeline_ids.pkl"  # optional: pipeline saved during training
DEFAULT_TRAIN_CSV = "data/UNSW_NB15_training-set.csv"


def load_model(path: str) -> object:
    """Load a model or pipeline from disk via joblib."""
    return joblib.load(path)


def _get_expected_feature_names(model: object) -> List[str]:
    """
    Attempt to discover the feature names the model expects.
    Tries several common locations (sklearn attribute, LightGBM booster,
    or pipeline final estimator).
    """
    # direct sklearn attribute
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # LightGBM booster
    try:
        if hasattr(model, "booster_"):
            return list(model.booster_.feature_name())
    except Exception:
        pass

    # pipeline: try to inspect the final estimator inside a sklearn Pipeline-like object
    try:
        if hasattr(model, "named_steps"):
            # get final estimator
            final = list(model.named_steps.values())[-1]
            if hasattr(final, "feature_names_in_"):
                return list(final.feature_names_in_)
            if hasattr(final, "booster_"):
                try:
                    return list(final.booster_.feature_name())
                except Exception:
                    pass
    except Exception:
        pass

    return []


def _malicious_prob_from_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    """
    Return the probability of the 'malicious' class robustly.
    Tries to find index for `1` in model.classes_. Falls back to last column or to mapping predict->0/1.
    """
    # If model has predict_proba, use it
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X))
        classes = list(getattr(model, "classes_", []))
        if probs.ndim == 1:
            # some models may return 1d probabilities
            return probs.astype(float)
        # try find class index for label 1
        try:
            if 1 in classes:
                idx = classes.index(1)
                return probs[:, idx].astype(float)
            if "MALICIOUS" in classes:
                idx = classes.index("MALICIOUS")
                return probs[:, idx].astype(float)
        except Exception:
            pass
        # fallback: return last column (common convention)
        return probs[:, -1].astype(float)
    else:
        # fallback: map predict -> probability 1 or 0
        preds = model.predict(X)
        return np.array([1.0 if int(p) == 1 else 0.0 for p in preds], dtype=float)


def _apply_train_scaling(df_aligned: pd.DataFrame, train_csv: str = DEFAULT_TRAIN_CSV) -> None:
    """
    If a training CSV is available, z-score each column in-place using training mean/std.
    This is a best-effort helper: it does nothing if the CSV or columns are missing.
    """
    try:
        if not os.path.exists(train_csv):
            return
        train_df = pd.read_csv(train_csv)
        # drop known meta columns
        for c in ("label", "attack_cat", "id"):
            if c in train_df.columns:
                train_df = train_df.drop(columns=[c], errors="ignore")
        stats = train_df.describe().T
        for col in df_aligned.columns:
            if col in stats.index:
                mu = stats.loc[col, "mean"]
                sd = stats.loc[col, "std"]
                sd = sd if (not pd.isna(sd) and sd > 0) else 1.0
                df_aligned[col] = (df_aligned[col].astype(float) - mu) / sd
    except Exception:
        # scaling is optional; swallow errors
        return


def align_pcap_df_to_model(df: pd.DataFrame, model: object, train_csv: str = DEFAULT_TRAIN_CSV) -> pd.DataFrame:
    """
    Align an extracted PCAP features DataFrame (e.g. columns f0..f41) to the model's expected feature names.
    Strategy:
      - If df already contains >=40% of the expected columns, use them (fill missing with 0).
      - Otherwise, map sequential f-columns (f0->expected[0], f1->expected[1], ...) as a bridge.
      - Optionally apply simple train-scaling if train_csv exists.
    Returns a new DataFrame with columns ordered as the model expects.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    expected = _get_expected_feature_names(model)
    if not expected:
        # no expected names found, return original df (caller will pad if needed)
        return df.copy()

    # quick check: how many expected columns are already present
    present = [c for c in expected if c in df.columns]
    if len(present) >= max(1, int(0.4 * len(expected))):  # heuristic threshold
        aligned = pd.DataFrame(0, index=df.index, columns=expected)
        for c in expected:
            if c in df.columns:
                aligned[c] = df[c].values
        _apply_train_scaling(aligned, train_csv)
        return aligned

    # fallback: map f-col names sequentially into expected names
    fcols = [c for c in df.columns if str(c).lower().startswith("f")]
    # if no fcols found, return zero-filled aligned frame (best-effort)
    aligned = pd.DataFrame(0, index=df.index, columns=expected)
    if not fcols:
        _apply_train_scaling(aligned, train_csv)
        return aligned

    # try sort fcols numerically (f0, f1, f2 ...)
    try:
        fcols_sorted = sorted(fcols, key=lambda x: int("".join(filter(str.isdigit, str(x))) or 0))
    except Exception:
        fcols_sorted = fcols

    for i, feat in enumerate(expected):
        if i < len(fcols_sorted):
            aligned[feat] = df[fcols_sorted[i]].values
        else:
            aligned[feat] = 0

    _apply_train_scaling(aligned, train_csv)
    return aligned


def predict_traffic(
    sample_data,
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
    benign_label: str = "BENIGN",
    malicious_label: str = "MALICIOUS",
    threshold: float = 0.5,
    train_csv: str = DEFAULT_TRAIN_CSV,
) -> List[Dict[str, Any]]:
    """
    Predict intrusion on the given data.
    - sample_data: pd.DataFrame or array-like of features (can be f0..fN or named columns)
    - model_path / pipeline_path: joblib paths
    - returns: list of dicts {'prediction': label, 'malicious_probability': float}
    """
    # load pipeline preferred, else raw model
    model = None
    loaded_from = None
    if os.path.exists(pipeline_path):
        try:
            model = load_model(pipeline_path)
            loaded_from = pipeline_path
        except Exception:
            model = None
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Neither pipeline ({pipeline_path}) nor model ({model_path}) found.")
        model = load_model(model_path)
        loaded_from = model_path

    # ensure DataFrame
    if not isinstance(sample_data, pd.DataFrame):
        sample_data = pd.DataFrame(sample_data)

    # determine expected features and align if needed
    expected = _get_expected_feature_names(model)
    X_in = sample_data.copy()

    if expected:
        # if there is no overlap between sample and expected, try adapter (map f0.. to expected)
        overlap = [c for c in X_in.columns if c in expected]
        if len(overlap) == 0:
            X_in = align_pcap_df_to_model(X_in, model, train_csv=train_csv)
        else:
            # make sure all expected columns exist in the right order
            aligned = pd.DataFrame(0, index=range(len(X_in)), columns=expected)
            for c in X_in.columns:
                if c in aligned.columns:
                    aligned[c] = X_in[c].values
            X_in = aligned

    # if no expected names discovered, we use the sample columns as-is
    # compute malicious probabilities
    probs = _malicious_prob_from_proba(model, X_in)

    results: List[Dict[str, Any]] = []
    for p in probs:
        label = malicious_label if (p is not None and p >= threshold) else benign_label
        results.append({"prediction": label, "malicious_probability": float(p if p is not None else 0.0)})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE_PATH)
    parser.add_argument("--csv", default=None, help="CSV of sample rows to predict")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv).head(args.n)
        # drop label if present
        for c in ("label", "attack_cat", "id"):
            if c in df.columns:
                df = df.drop(columns=[c], errors="ignore")
    else:
        # minimal dummy sample
        df = pd.DataFrame(np.random.rand(args.n, 42), columns=[f"f{i}" for i in range(42)])

    res = predict_traffic(df, model_path=args.model, pipeline_path=args.pipeline, threshold=args.threshold)
    for r in res:
        print(r)