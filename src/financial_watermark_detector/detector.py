from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .watermarking import FinancialWatermarkEngine, tokenize_words


@dataclass
class WatermarkTrainingResult:
    accuracy: float
    macro_f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    classification_report: dict


class WatermarkFeatureExtractor:
    def __init__(self, watermark_engine: FinancialWatermarkEngine):
        self.watermark_engine = watermark_engine
        self.feature_names = [
            "token_count",
            "candidate_count",
            "preferred_hit_count",
            "preferred_ratio",
            "z_score",
            "ticker_count",
            "digit_ratio",
            "uppercase_ratio",
            "punctuation_ratio",
            "avg_token_length",
        ]

    def transform(self, texts) -> np.ndarray:
        rows = []
        for text in texts:
            text = str(text)
            stats = self.watermark_engine.detect_statistics(text)
            raw_tokens = tokenize_words(text)
            rows.append(
                [
                    stats["token_count"],
                    stats["candidate_count"],
                    stats["preferred_hit_count"],
                    stats["preferred_ratio"],
                    stats["z_score"],
                    text.count("$"),
                    sum(char.isdigit() for char in text) / max(len(text), 1),
                    sum(char.isupper() for char in text) / max(len(text), 1),
                    sum(not char.isalnum() and not char.isspace() for char in text) / max(len(text), 1),
                    float(np.mean([len(token) for token in raw_tokens])) if raw_tokens else 0.0,
                ]
            )
        return np.array(rows, dtype=float)


class WatermarkDetector:
    def __init__(self, watermark_engine: FinancialWatermarkEngine, random_state: int = 42):
        self.watermark_engine = watermark_engine
        self.feature_extractor = WatermarkFeatureExtractor(watermark_engine)
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        )
        self.random_state = random_state

    def prepare_data(self, df: pd.DataFrame, text_col: str = "text", target_col: str = "label", test_size: float = 0.2):
        dataset = df[[text_col, target_col]].dropna().copy()
        X = self.feature_extractor.transform(dataset[text_col])
        y = dataset[target_col].astype(int).values
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> WatermarkTrainingResult:
        preds = self.predict(X_test)
        proba = self.predict_proba(X_test)[:, 1]
        return WatermarkTrainingResult(
            accuracy=accuracy_score(y_test, preds),
            macro_f1=f1_score(y_test, preds, average="macro"),
            roc_auc=roc_auc_score(y_test, proba),
            confusion_matrix=confusion_matrix(y_test, preds).tolist(),
            classification_report=classification_report(y_test, preds, output_dict=True),
        )

    def explain_feature_importance(self) -> pd.DataFrame:
        classifier = self.model.named_steps["classifier"]
        return pd.DataFrame(
            {"feature": self.feature_extractor.feature_names, "weight": classifier.coef_[0]}
        ).sort_values("weight", ascending=False)

    def plot_feature_importance(self, output_path: str):
        importance = self.explain_feature_importance().sort_values("weight")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(importance["feature"], importance["weight"], color="#1f77b4")
        ax.set_title("Watermark Detection Feature Importance")
        ax.set_xlabel("Coefficient")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
