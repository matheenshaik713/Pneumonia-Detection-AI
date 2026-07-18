"""
Image preprocessing utilities.

Author: Matheen Shaik
"""

from pathlib import Path
import cv2
import numpy as np

IMAGE_SIZE = (224, 224)


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    image_path : str | Path

    Returns
    -------
    np.ndarray
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_image(image: np.ndarray) -> np.ndarray:
    """
    Resize image to CNN input size.
    """

    return cv2.resize(image, IMAGE_SIZE)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixels to [0,1]
    """

    image = image.astype(np.float32)

    return image / 255.0


def preprocess_image(image_path: str | Path):

    image = load_image(image_path)

    image = resize_image(image)

    image = normalize_image(image)

    return image