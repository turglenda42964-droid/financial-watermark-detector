from src.financial_watermark_detector.watermarking import FinancialWatermarkEngine


def test_embed_watermark_returns_stats():
    engine = FinancialWatermarkEngine(secret_key="test-key")
    watermarked, stats = engine.embed_watermark("Strong revenue growth supports market expansion.")

    assert isinstance(watermarked, str)
    assert "candidate_count" in stats
    assert "watermark_ratio" in stats


def test_detect_statistics_returns_expected_fields():
    engine = FinancialWatermarkEngine(secret_key="test-key")
    stats = engine.detect_statistics("Robust turnover supports trading expansion.")

    assert "preferred_ratio" in stats
    assert "z_score" in stats
