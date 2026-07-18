"""
Model Training Module

Author: Matheen Shaik
"""

import tensorflow as tf

from src.model import PneumoniaCNN
from src.dataloader import DataLoader
from src.callbacks import get_callbacks

from configs.config import (
    LEARNING_RATE,
    EPOCHS,
    MODELS_DIR,
    IMAGE_SIZE
)


class Trainer:

    def __init__(self):

        self.loader = DataLoader()

        self.train_ds = self.loader.get_train_dataset()

        self.val_ds = self.loader.get_validation_dataset()

        self.model = PneumoniaCNN.build()

        # ====================================================
        # IMPORTANT FOR KERAS 3 + GRADCAM
        # Explicitly build the model before training
        # ====================================================

        self.model.build(
            (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )

        # OR (either one is enough)
        dummy = tf.random.normal(
            (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )

        _ = self.model(dummy)

        # ====================================================

        self.history = None

    def compile(self):

        self.model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            ),

            loss=tf.keras.losses.BinaryCrossentropy(),

            metrics=[

                tf.keras.metrics.BinaryAccuracy(
                    name="accuracy"
                ),

                tf.keras.metrics.Precision(
                    name="precision"
                ),

                tf.keras.metrics.Recall(
                    name="recall"
                )

            ]

        )

    def summary(self):

        self.model.summary()

    def train(self):

        self.history = self.model.fit(

            self.train_ds,

            validation_data=self.val_ds,

            epochs=EPOCHS,

            callbacks=get_callbacks(),

            verbose=1

        )

        return self.history

    def save(self):

        MODELS_DIR.mkdir(
            exist_ok=True,
            parents=True
        )

        self.model.save(
            MODELS_DIR / "final_model.keras"
        )

        print("\nModel Saved Successfully!")

    def run(self):

        self.compile()

        self.summary()

        self.train()

        self.save()