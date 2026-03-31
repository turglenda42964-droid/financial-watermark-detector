from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

#1.统一数据分析的语法糖（统一格式）
@dataclass
class CorpusSummary:
    num_rows: int
    num_columns: int
    text_column: str
    metadata_columns: list[str]
    sample_lengths: dict[str, float]

class FinancialTextDataLoader:
    """Load financial text corpus from `data/`."""
    
#2.选定/初始化项目路径
    def __init__(self, project_root: Optional[Path | str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.data_dir = self.project_root / "data"
        self.data: Optional[pd.DataFrame] = None

    def resolve_data_path(self, filename: Optional[str] = None) -> Path:
        candidates = []
        if filename:
            candidates.append(self.data_dir / filename)
        candidates.extend([self.data_dir / "data.csv", self.data_dir / "data.cvs"])
        for path in candidates:
            if path.exists():
                return path
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(f"Missing dataset. Expected one of: {names}")
        
#3.定义数据类型，数据规范化清洗
    def load_corpus(self, filename: Optional[str] = None, text_column: str = "Sentence") -> pd.DataFrame:
        data_path = self.resolve_data_path(filename)
        df = pd.read_csv(data_path)
        if text_column not in df.columns:
            raise ValueError(f"Missing required text column: {text_column}")
        df = df.copy()
        df[text_column] = df[text_column].astype(str).str.strip()
        df = df.dropna(subset=[text_column])
        df = df[df[text_column] != ""]
        df = df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
        self.data = df
        return df
#4.数据统计
    
    def summarize_corpus(self, df: Optional[pd.DataFrame] = None, text_column: str = "Sentence") -> CorpusSummary:
        corpus = df if df is not None else self.data
        if corpus is None:
            raise ValueError("Call load_corpus() first.")
        lengths = corpus[text_column].astype(str).str.split().str.len()
        return CorpusSummary(
            num_rows=len(corpus),
            num_columns=corpus.shape[1],
            text_column=text_column,
            metadata_columns=[col for col in corpus.columns if col != text_column],
            sample_lengths={
                "mean_tokens": float(lengths.mean()),
                "median_tokens": float(lengths.median()),
                "max_tokens": float(lengths.max()),
            },
        )
#总结：适用于全部文本/情绪分析的数据类型，但黑河运行，缺少项目日志
