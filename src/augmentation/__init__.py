"""
Data augmentation utilities used during training.
Currently exposes SMOTE oversampling and Focal Loss — both defined here
so other modules can import from a single place.
"""

from imblearn.over_sampling import SMOTE
import numpy as np


def apply_smote(X: np.ndarray, y: np.ndarray,
                sampling_strategy: float = 0.1,
                k_neighbors: int = 5,
                random_state: int = 42):
    """
    Oversample the minority (fraud) class using SMOTE.
    Only applied to the training split, never to val/test.

    sampling_strategy: desired ratio of minority to majority after resampling.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    return smote.fit_resample(X, y)
