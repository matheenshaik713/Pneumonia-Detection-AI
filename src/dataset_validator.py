"""
Dataset Validation Module

Author: Matheen Shaik

Description:
------------
This module validates the Chest X-ray dataset before training.

Features:
---------
1. Counts images
2. Detects corrupted images
3. Checks file extensions
4. Displays class distribution
"""

from pathlib import Path
import cv2
import logging
from collections import defaultdict

# Supported image formats
VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class DatasetValidator:

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

        self.image_count = defaultdict(int)

        self.corrupted_images = []

        self.invalid_files = []

    def validate(self):

        logging.info("Starting Dataset Validation...")

        for split in ["train", "val", "test"]:

            split_path = self.dataset_path / split

            if not split_path.exists():

                raise FileNotFoundError(
                    f"{split} folder not found."
                )

            for class_folder in split_path.iterdir():

                if not class_folder.is_dir():
                    continue

                class_name = class_folder.name

                for image_path in class_folder.iterdir():

                    if image_path.suffix.lower() not in VALID_EXTENSIONS:

                        self.invalid_files.append(str(image_path))

                        continue

                    image = cv2.imread(str(image_path))

                    if image is None:

                        self.corrupted_images.append(str(image_path))

                        continue

                    self.image_count[
                        f"{split}/{class_name}"
                    ] += 1

        logging.info("Validation Completed.")

    def summary(self):

        print("\n")
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        total = 0

        for key, value in self.image_count.items():

            print(f"{key:<25} : {value}")

            total += value

        print("-" * 60)
        print(f"Total Images : {total}")
        print("-" * 60)

        print(f"Corrupted Images : {len(self.corrupted_images)}")

        print(f"Invalid Files : {len(self.invalid_files)}")

        print("=" * 60)

    def show_errors(self):

        if self.corrupted_images:

            print("\nCorrupted Images")

            for img in self.corrupted_images:

                print(img)

        if self.invalid_files:

            print("\nInvalid Files")

            for img in self.invalid_files:

                print(img)