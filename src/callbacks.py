"""
Training callbacks.

Author: Matheen Shaik
"""

from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)

from configs.config import (
    MODELS_DIR,
    LOGS_DIR
)


def get_callbacks():
    """
    Create all callbacks used during training.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            verbose=1
        ),

        CSVLogger(
            str(LOGS_DIR / "training_log.csv")
        ),

        TensorBoard(
            log_dir=str(LOGS_DIR / "tensorboard")
        )

    ]

    return callbacks