"""
Data augmentation pipeline
"""

import albumentations as A

train_transform = A.Compose([

    A.HorizontalFlip(p=0.5),

    A.Rotate(limit=10, p=0.5),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    A.ShiftScaleRotate(

        shift_limit=0.05,

        scale_limit=0.05,

        rotate_limit=10,

        p=0.5
    )

])

validation_transform = A.Compose([])