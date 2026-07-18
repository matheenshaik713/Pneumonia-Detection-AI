"""
Prediction Module

Author: Matheen Shaik

Predict Pneumonia from a single Chest X-ray.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocess import preprocess_image
from configs.config import (
    MODELS_DIR,
    CLASS_NAMES
)


class Predictor:

    def __init__(self):

        self.model = tf.keras.models.load_model(
            MODELS_DIR / "best_model.keras"
        )

    def predict(self, image_path):

        # Preprocess image
        image = preprocess_image(image_path)

        image = np.expand_dims(image, axis=0)

        # Predict probability
        probability = float(
            self.model.predict(
                image,
                verbose=0
            )[0][0]
        )

        # Convert probability to class index
        prediction = 1 if probability >= 0.5 else 0

        # Confidence score
        confidence = (
            probability if prediction == 1
            else 1 - probability
        ) * 100

        # Convert index to class label
        prediction_label = CLASS_NAMES[prediction]

        # Display result
        print("\n" + "=" * 60)
        print("PREDICTION")
        print("=" * 60)
        print("Image      :", Path(image_path).name)
        print("Prediction :", prediction_label)
        print(f"Confidence : {confidence:.2f}%")
        print("=" * 60)

        # Return for FastAPI/Streamlit
        return prediction_label, float(confidence)