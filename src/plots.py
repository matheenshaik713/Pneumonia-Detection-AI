"""
Visualization Module

Author: Matheen Shaik

Generates:

1. Accuracy Curve
2. Loss Curve
3. Confusion Matrix Heatmap
4. ROC Curve
5. Precision-Recall Curve
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    precision_recall_curve
)

from configs.config import (
    MODELS_DIR,
    REPORTS_DIR
)

from src.dataloader import DataLoader


class PlotGenerator:

    def __init__(self):

        REPORTS_DIR.mkdir(exist_ok=True)

        self.loader = DataLoader()

        self.test_ds = self.loader.get_test_dataset()

        self.model = tf.keras.models.load_model(
            MODELS_DIR / "best_model.keras"
        )

    def plot_history(self):

        csv_path = Path("logs/training_log.csv")

        import pandas as pd

        history = pd.read_csv(csv_path)

        plt.figure(figsize=(8,5))

        plt.plot(history["accuracy"], label="Train Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")

        plt.legend()

        plt.savefig(REPORTS_DIR / "accuracy_curve.png")

        plt.close()

        plt.figure(figsize=(8,5))

        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.legend()

        plt.savefig(REPORTS_DIR / "loss_curve.png")

        plt.close()

    def plot_confusion_matrix(self):

        y_true = []
        y_pred = []

        for images, labels in self.test_ds:

            preds = self.model.predict(images, verbose=0)

            preds = (preds > 0.5).astype(int)

            y_true.extend(labels.numpy())

            y_pred.extend(preds.flatten())

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["NORMAL","PNEUMONIA"]
        )

        disp.plot(cmap="Blues")

        plt.title("Confusion Matrix")

        plt.savefig(REPORTS_DIR / "confusion_matrix.png")

        plt.close()

    def plot_roc(self):

        y_true = []
        y_score = []

        for images, labels in self.test_ds:

            preds = self.model.predict(images, verbose=0)

            y_true.extend(labels.numpy())

            y_score.extend(preds.flatten())

        fpr, tpr, _ = roc_curve(y_true, y_score)

        RocCurveDisplay(
            fpr=fpr,
            tpr=tpr
        ).plot()

        plt.savefig(REPORTS_DIR / "roc_curve.png")

        plt.close()

    def plot_precision_recall(self):

        y_true = []
        y_score = []

        for images, labels in self.test_ds:

            preds = self.model.predict(images, verbose=0)

            y_true.extend(labels.numpy())

            y_score.extend(preds.flatten())

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_score
        )

        PrecisionRecallDisplay(
            precision=precision,
            recall=recall
        ).plot()

        plt.savefig(REPORTS_DIR / "precision_recall_curve.png")

        plt.close()

    def generate(self):

        self.plot_history()

        self.plot_confusion_matrix()

        self.plot_roc()

        self.plot_precision_recall()

        print("\nAll plots generated successfully!")