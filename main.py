from __future__ import annotations

from pathlib import Path

from src.financial_watermark_detector import run_pipeline


if __name__ == "__main__":
    outputs = run_pipeline(Path(__file__).resolve().parent)
    print("Financial watermark detector pipeline completed.")
    for key, value in outputs.items():
        print(f"{key}: {value}")
