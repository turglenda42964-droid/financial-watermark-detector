from __future__ import annotations

import json
from pathlib import Path

import joblib

from .data import FinancialTextDataLoader
from .detector import WatermarkDetector
from .watermarking import FinancialWatermarkEngine, WatermarkDatasetBuilder


def run_pipeline(project_root: Path | str | None = None) -> dict[str, str]:
    project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    for directory in [models_dir, reports_dir, data_dir, results_dir / "figures", results_dir / "reports"]:
        directory.mkdir(parents=True, exist_ok=True)

    loader = FinancialTextDataLoader(project_root=project_root)
    corpus = loader.load_corpus()
    summary = loader.summarize_corpus(corpus)

    watermark_engine = FinancialWatermarkEngine()
    dataset_builder = WatermarkDatasetBuilder(watermark_engine)
    watermark_df = dataset_builder.build(corpus)

    detector = WatermarkDetector(watermark_engine)
    X_train, X_test, y_train, y_test = detector.prepare_data(watermark_df)
    detector.train(X_train, y_train)
    metrics = detector.evaluate(X_test, y_test)

    outputs = {
        "model": str(models_dir / "watermark_detector.joblib"),
        "dataset": str(data_dir / "watermark_detection_dataset.csv"),
        "corpus_summary": str(reports_dir / "corpus_summary.json"),
        "metrics": str(reports_dir / "watermark_metrics.json"),
        "feature_plot": str(reports_dir / "watermark_feature_importance.png"),
        "samples": str(reports_dir / "watermark_samples.json"),
    }

    joblib.dump({"watermark_engine": watermark_engine, "detector": detector}, outputs["model"])
    watermark_df.to_csv(outputs["dataset"], index=False)
    detector.plot_feature_importance(outputs["feature_plot"])

    sample_rows = []
    for sample in watermark_df[watermark_df["label"] == 0]["text"].head(5).tolist():
        watermarked_text, embed_stats = watermark_engine.embed_watermark(sample)
        detect_stats = watermark_engine.detect_statistics(watermarked_text)
        sample_rows.append(
            {
                "original_text": sample,
                "watermarked_text": watermarked_text,
                "embed_stats": embed_stats,
                "detect_stats": detect_stats,
            }
        )

    with open(outputs["corpus_summary"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_rows": summary.num_rows,
                "num_columns": summary.num_columns,
                "text_column": summary.text_column,
                "metadata_columns": summary.metadata_columns,
                "sample_lengths": summary.sample_lengths,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(outputs["metrics"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "roc_auc": metrics.roc_auc,
                "confusion_matrix": metrics.confusion_matrix,
                "classification_report": metrics.classification_report,
                "feature_importance": detector.explain_feature_importance().to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(outputs["samples"], "w", encoding="utf-8") as f:
        json.dump(sample_rows, f, ensure_ascii=False, indent=2)

    return outputs
