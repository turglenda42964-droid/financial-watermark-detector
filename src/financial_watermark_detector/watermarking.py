from __future__ import annotations

import hashlib
import re

import pandas as pd


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\$?[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text)


class FinancialWatermarkEngine:
    """A lightweight lexical watermark engine for financial text."""

    def __init__(self, secret_key: str = "fin-watermark-2026"):
        self.secret_key = secret_key
        self.variant_groups = [
            ("rise", "climb"),
            ("falls", "drops"),
            ("fall", "drop"),
            ("gain", "advance"),
            ("gains", "advances"),
            ("profit", "earnings"),
            ("profits", "earnings"),
            ("share", "equity"),
            ("shares", "equities"),
            ("market", "trading"),
            ("stock", "security"),
            ("stocks", "securities"),
            ("investor", "holder"),
            ("investors", "holders"),
            ("growth", "expansion"),
            ("risk", "exposure"),
            ("forecast", "outlook"),
            ("strong", "robust"),
            ("weak", "soft"),
            ("increase", "boost"),
            ("decrease", "trim"),
            ("revenue", "turnover"),
            ("bank", "lender"),
            ("banks", "lenders"),
        ]
        self.lookup = {}
        self.preferred_tokens = set()
        for original, preferred in self.variant_groups:
            self.lookup[original] = preferred
            self.lookup[preferred] = preferred
            self.preferred_tokens.add(preferred)

    def _greenlist_hit(self, token: str, position: int) -> bool:
        digest = hashlib.sha256(f"{self.secret_key}:{position}:{token.lower()}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 2 == 0

    def embed_watermark(self, text: str) -> tuple[str, dict[str, float]]:
        candidates = 0
        green_hits = 0
        position = 0

        def replace(match: re.Match) -> str:
            nonlocal candidates, green_hits, position
            piece = match.group(0)
            lowered = piece.lower()
            current_position = position
            position += 1
            if lowered not in self.lookup:
                return piece
            candidates += 1
            if not self._greenlist_hit(lowered, current_position):
                return piece
            replacement = self.lookup[lowered]
            if piece.istitle():
                replacement = replacement.title()
            elif piece.isupper():
                replacement = replacement.upper()
            green_hits += 1
            return replacement

        watermarked = re.sub(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", replace, text)
        ratio = green_hits / candidates if candidates else 0.0
        return watermarked, {
            "candidate_count": float(candidates),
            "green_hit_count": float(green_hits),
            "watermark_ratio": float(ratio),
        }

    def detect_statistics(self, text: str) -> dict[str, float]:
        tokens = tokenize_words(text.lower())
        candidates = [token for token in tokens if token in self.lookup]
        preferred_hits = sum(token in self.preferred_tokens for token in candidates)
        candidate_count = len(candidates)
        ratio = preferred_hits / candidate_count if candidate_count else 0.0
        expected = candidate_count * 0.5
        variance = candidate_count * 0.25
        z_score = (preferred_hits - expected) / (variance ** 0.5) if variance > 0 else 0.0
        return {
            "token_count": float(len(tokens)),
            "candidate_count": float(candidate_count),
            "preferred_hit_count": float(preferred_hits),
            "preferred_ratio": float(ratio),
            "z_score": float(z_score),
        }


class WatermarkDatasetBuilder:
    """Create paired original/watermarked samples."""

    def __init__(self, watermark_engine: FinancialWatermarkEngine):
        self.watermark_engine = watermark_engine

    def build(self, df: pd.DataFrame, text_col: str = "Sentence") -> pd.DataFrame:
        rows = []
        for source_text in df[text_col].astype(str):
            clean_text = source_text.strip()
            if len(clean_text.split()) < 4:
                continue
            watermarked_text, stats = self.watermark_engine.embed_watermark(clean_text)
            if stats["candidate_count"] <= 0 or watermarked_text == clean_text:
                continue
            rows.append(
                {
                    "text": clean_text,
                    "label": 0,
                    "source_type": "original",
                    "watermark_ratio": 0.0,
                    "candidate_count": stats["candidate_count"],
                }
            )
            rows.append(
                {
                    "text": watermarked_text,
                    "label": 1,
                    "source_type": "watermarked",
                    "watermark_ratio": stats["watermark_ratio"],
                    "candidate_count": stats["candidate_count"],
                }
            )
        return pd.DataFrame(rows)
