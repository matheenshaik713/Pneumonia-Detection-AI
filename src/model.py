"""
Custom CNN Model

Author: Matheen Shaik

Description
-----------
Defines a production-ready CNN architecture for
Pneumonia Detection.
"""

from keras import Sequential
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from configs.config import (
    IMAGE_SIZE,
    DROPOUT_RATE,
    NUM_CLASSES
)


class PneumoniaCNN:

    @staticmethod
    def build():

        model = Sequential(

            [

                Input(shape=(*IMAGE_SIZE, 3)),

                Conv2D(
                    32,
                    (3, 3),
                    padding="same"
                ),

                BatchNormalization(),

                ReLU(),

                MaxPooling2D(),


                Conv2D(
                    64,
                    (3, 3),
                    padding="same"
                ),

                BatchNormalization(),

                ReLU(),

                MaxPooling2D(),


                Conv2D(
                    128,
                    (3, 3),
                    padding="same"
                ),

                BatchNormalization(),

                ReLU(),

                MaxPooling2D(),


                Conv2D(
                    256,
                    (3, 3),
                    padding="same"
                ),

                BatchNormalization(),

                ReLU(),

                GlobalAveragePooling2D(),

                Dropout(DROPOUT_RATE),

                Dense(
                    128,
                    activation="relu"
                ),

                Dropout(DROPOUT_RATE),

                Dense(
                    NUM_CLASSES,
                    activation="sigmoid"
                )

            ]

        )

        return model