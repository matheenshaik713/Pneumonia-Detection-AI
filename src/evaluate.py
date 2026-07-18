"""
Model Evaluation

Author: Matheen Shaik
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.dataloader import DataLoader
from configs.config import MODELS_DIR


class Evaluator:

    def __init__(self):

        self.loader = DataLoader()

        self.test_ds = self.loader.get_test_dataset()

        self.model = tf.keras.models.load_model(
            MODELS_DIR / "best_model.keras"
        )

    def evaluate(self):

        y_true = []
        y_pred = []

        for images, labels in self.test_ds:

            predictions = self.model.predict(images, verbose=0)

            predictions = (predictions > 0.5).astype(int)

            y_true.extend(labels.numpy().astype(int))

            y_pred.extend(predictions.flatten())

        print("\n")
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(
            "Accuracy :",
            round(accuracy_score(y_true, y_pred), 4)
        )

        print(
            "Precision:",
            round(precision_score(y_true, y_pred), 4)
        )

        print(
            "Recall   :",
            round(recall_score(y_true, y_pred), 4)
        )

        print(
            "F1 Score :",
            round(f1_score(y_true, y_pred), 4)
        )

        print("\nClassification Report\n")

        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["NORMAL", "PNEUMONIA"]
            )
        )

        print("\nConfusion Matrix\n")

        print(confusion_matrix(y_true, y_pred))