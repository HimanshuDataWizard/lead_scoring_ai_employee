from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lead_scoring.train import train_and_save


def main() -> None:
    result = train_and_save()
    print(f"Best model: {result.best_model_name}")
    print("Saved artifacts to ./artifacts")
    print("Saved report: ./artifacts/model_comparison.md")
    print("Key metrics:")
    print(f"  ROC-AUC: {result.metrics['roc_auc']}")
    print(f"  F1: {result.metrics['f1']}")
    print(f"  Precision: {result.metrics['precision']}")
    print(f"  Recall: {result.metrics['recall']}")
    print(f"  Threshold: {result.metrics['decision_threshold']}")
    print(f"  Calibration: {result.metrics['model']['calibration']}")


if __name__ == "__main__":
    main()
