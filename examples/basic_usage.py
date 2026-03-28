from pathlib import Path

from src.financial_watermark_detector import FinancialTextDataLoader, FinancialWatermarkEngine


project_root = Path(__file__).resolve().parents[1]

loader = FinancialTextDataLoader(project_root=project_root)
corpus = loader.load_corpus()
sample_text = corpus["Sentence"].iloc[0]

engine = FinancialWatermarkEngine()
watermarked_text, embed_stats = engine.embed_watermark(sample_text)
detect_stats = engine.detect_statistics(watermarked_text)

print("Original:")
print(sample_text)
print("\nWatermarked:")
print(watermarked_text)
print("\nEmbed stats:", embed_stats)
print("Detect stats:", detect_stats)
