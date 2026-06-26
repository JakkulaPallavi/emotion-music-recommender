"""
evaluate.py
------------
Evaluates the emotion detection model with cross-validation and
prints a detailed classification report.

Run with:
    python src/evaluate.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from emotion_detector import TRAINING_DATA, build_model, clean_text


def evaluate():
    texts, labels = zip(*TRAINING_DATA)
    texts  = list(texts)
    labels = list(labels)

    model = build_model()

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, texts, labels, cv=cv, scoring="accuracy")

    print("=" * 55)
    print("  Emotion Detector — Model Evaluation Report")
    print("=" * 55)
    print(f"\nDataset size : {len(texts)} samples across {len(set(labels))} emotions")
    print(f"\nCross-Validation Accuracy (5-fold):")
    print(f"  Per fold : {[f'{s:.2%}' for s in scores]}")
    print(f"  Mean     : {scores.mean():.2%}")
    print(f"  Std Dev  : {scores.std():.2%}")

    # Train on full set and print classification report
    model.fit(texts, labels)
    preds = model.predict(texts)

    print("\nClassification Report (trained on full dataset):")
    print(classification_report(labels, preds, zero_division=0))

    print("\nConfusion Matrix:")
    classes = sorted(set(labels))
    cm = confusion_matrix(labels, preds, labels=classes)
    header = "         " + "  ".join(f"{c[:5]:>5}" for c in classes)
    print(header)
    for row_label, row in zip(classes, cm):
        print(f"  {row_label:<8} " + "  ".join(f"{v:>5}" for v in row))

    print("\nNote: These scores are on TRAINING data.")
    print("For real-world performance, add more diverse samples to TRAINING_DATA.")


if __name__ == "__main__":
    evaluate()
