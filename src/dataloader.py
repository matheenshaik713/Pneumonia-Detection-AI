"""
Production TensorFlow Data Loader

Author: Matheen Shaik

Description
-----------
Loads train, validation and test datasets using
TensorFlow's tf.data API.
"""

from pathlib import Path
import tensorflow as tf

from configs.config import (
    RAW_DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_STATE
)


class DataLoader:
    """
    Production Data Loader
    """

    def __init__(self):

        self.dataset_path = Path(RAW_DATA_DIR) / "chest_xray"

        self.train_path = self.dataset_path / "train"

        self.val_path = self.dataset_path / "val"

        self.test_path = self.dataset_path / "test"

    def _load_dataset(self, path: Path, shuffle: bool):

        dataset = tf.keras.utils.image_dataset_from_directory(

            path,

            labels="inferred",

            label_mode="binary",

            image_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            shuffle=shuffle,

            seed=RANDOM_STATE

        )

        return dataset

    @staticmethod
    def optimize(dataset):

        AUTOTUNE = tf.data.AUTOTUNE

        dataset = dataset.cache()

        dataset = dataset.prefetch(AUTOTUNE)

        return dataset

    def get_train_dataset(self):

        dataset = self._load_dataset(

            self.train_path,

            shuffle=True

        )

        return self.optimize(dataset)

    def get_validation_dataset(self):

        dataset = self._load_dataset(

            self.val_path,

            shuffle=False

        )

        return self.optimize(dataset)

    def get_test_dataset(self):

        dataset = self._load_dataset(

            self.test_path,

            shuffle=False

        )

        return self.optimize(dataset)

    def get_class_names(self):

        dataset = self._load_dataset(

            self.train_path,

            shuffle=False

        )

        return dataset.class_names