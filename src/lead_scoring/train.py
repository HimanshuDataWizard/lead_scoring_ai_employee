from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    ID_COLUMN,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

COMPARISON_REPORT_PATH = ARTIFACTS_DIR / "model_comparison.md"


@dataclass
class TrainResult:
    best_model_name: str
    metrics: Dict[str, Any]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates lightweight domain-inspired features from base columns."""

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        frame["deal_value_log"] = np.log1p(frame["deal_value"])
        frame["company_size_log"] = np.log1p(frame["company_size"])
        frame["purchase_intensity"] = frame["prev_purchases"] / (frame["response_time"] + 1.0)
        frame["contact_freshness"] = 1.0 / (frame["last_contact"] + 1.0)
        frame["deal_per_employee"] = frame["deal_value"] / (frame["company_size"] + 1.0)
        frame["response_contact_product"] = frame["response_time"] * frame["last_contact"]
        frame["deal_purchase_product"] = frame["deal_value"] * (frame["prev_purchases"] + 1.0)

        frame["industry_source"] = frame["industry"].astype(str) + "__" + frame["source"].astype(str)
        frame["region_source"] = frame["region"].astype(str) + "__" + frame["source"].astype(str)

        return frame


def _feature_lists() -> Tuple[List[str], List[str]]:
    engineered_numeric = [
        "deal_value_log",
        "company_size_log",
        "purchase_intensity",
        "contact_freshness",
        "deal_per_employee",
        "response_contact_product",
        "deal_purchase_product",
    ]
    engineered_categorical = ["industry_source", "region_source"]
    numeric = NUMERIC_FEATURES + engineered_numeric
    categorical = CATEGORICAL_FEATURES + engineered_categorical
    return numeric, categorical


def _build_preprocessor() -> ColumnTransformer:
    numeric_features, categorical_features = _feature_lists()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def _candidate_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=5000, class_weight="balanced"),
            "params": {
                "model__C": [0.1, 1.0, 5.0],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            "params": {
                "model__n_estimators": [250, 500],
                "model__max_depth": [None, 10, 16],
                "model__min_samples_split": [2, 6],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt"],
            },
        },
        "extra_trees": {
            "estimator": ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            "params": {
                "model__n_estimators": [250, 500],
                "model__max_depth": [None, 10, 16],
                "model__min_samples_split": [2, 6],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt"],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "model__n_estimators": [100, 180],
                "model__learning_rate": [0.03, 0.07, 0.12],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        },
    }


def _binary_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def _optimize_threshold(y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    candidates = np.arange(0.2, 0.81, 0.01)

    best_threshold = 0.5
    best_score = -1.0
    best_precision = 0.0

    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        if score > best_score or (score == best_score and precision > best_precision):
            best_score = float(score)
            best_precision = float(precision)
            best_threshold = float(threshold)

    return best_threshold, {
        "best_f1": round(best_score, 4),
        "best_precision": round(best_precision, 4),
    }


def _write_comparison_report(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "# Model Comparison Report (V3)",
        "",
        "| Model | Best CV ROC-AUC | Validation ROC-AUC | Validation Brier |",
        "|---|---:|---:|---:|",
    ]

    sorted_rows = sorted(rows, key=lambda r: (r["val_roc_auc"], -r["val_brier"]), reverse=True)
    for row in sorted_rows:
        lines.append(
            f"| {row['model_name']} | {row['cv_roc_auc']:.4f} | {row['val_roc_auc']:.4f} | {row['val_brier']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_and_save() -> TrainResult:
    df = pd.read_csv(DATA_PATH)

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    preprocessor = _build_preprocessor()
    feature_engineer = FeatureEngineer()
    candidates = _candidate_configs()

    comparison_rows: List[Dict[str, Any]] = []
    best_name = ""
    best_params: Dict[str, Any] = {}
    best_estimator: Any = None
    best_val_auc = -1.0

    for name, cfg in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("feature_engineer", feature_engineer),
                ("preprocessor", preprocessor),
                ("model", cfg["estimator"]),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=cfg["params"],
            scoring="roc_auc",
            cv=5,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)

        tuned = search.best_estimator_
        val_prob = tuned.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        val_brier = brier_score_loss(y_val, val_prob)

        comparison_rows.append(
            {
                "model_name": name,
                "cv_roc_auc": float(search.best_score_),
                "val_roc_auc": float(val_auc),
                "val_brier": float(val_brier),
            }
        )

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_name = name
            best_params = search.best_params_
            best_estimator = tuned

    if best_estimator is None:
        raise RuntimeError("No model candidate could be trained.")

    calibrated_candidates: List[Tuple[str, Any, np.ndarray]] = []
    calibrated_candidates.append(("none", best_estimator, best_estimator.predict_proba(X_val)[:, 1]))

    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(estimator=clone(best_estimator), method=method, cv=5)
        calibrated.fit(X_train, y_train)
        calibrated_candidates.append((method, calibrated, calibrated.predict_proba(X_val)[:, 1]))

    best_calibration_method = "none"
    best_calibration_model: Any = best_estimator
    best_calibration_brier = brier_score_loss(y_val, best_estimator.predict_proba(X_val)[:, 1])

    for method, model, val_prob in calibrated_candidates:
        brier = brier_score_loss(y_val, val_prob)
        if brier < best_calibration_brier:
            best_calibration_brier = float(brier)
            best_calibration_method = method
            best_calibration_model = model

    val_prob_best = best_calibration_model.predict_proba(X_val)[:, 1]
    threshold, threshold_info = _optimize_threshold(y_val, val_prob_best)

    # Refit final model on train+validation with chosen model params and calibration strategy.
    final_pipeline = clone(best_estimator)
    final_pipeline.fit(X_train_val, y_train_val)

    if best_calibration_method == "none":
        final_model = final_pipeline
    else:
        final_model = CalibratedClassifierCV(estimator=clone(final_pipeline), method=best_calibration_method, cv=5)
        final_model.fit(X_train_val, y_train_val)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    final_metrics = _binary_metrics(y_test, test_prob, threshold)

    _write_comparison_report(comparison_rows, COMPARISON_REPORT_PATH)

    metrics: Dict[str, Any] = {
        **final_metrics,
        "decision_threshold": round(threshold, 4),
        "threshold_selection": threshold_info,
        "model": {
            "name": best_name,
            "best_params": best_params,
            "validation_roc_auc": round(best_val_auc, 4),
            "calibration": best_calibration_method,
        },
        "comparison": comparison_rows,
        "dataset": {
            "rows": int(df.shape[0]),
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "split": {
                "train": int(X_train.shape[0]),
                "validation": int(X_val.shape[0]),
                "test": int(X_test.shape[0]),
            },
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainResult(best_model_name=best_name, metrics=metrics)
