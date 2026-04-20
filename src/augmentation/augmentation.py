"""
Training-time data augmentation utilities.

Two strategies are used across the platform:

  SMOTE (XGBoost)  — synthetic minority oversampling generates new fraud
                     examples by interpolating between real ones.  Applied
                     to the training split only; val and test are never
                     touched so metrics reflect real-world class balance.

  Focal Loss (GAT) — instead of oversampling, the loss function itself
                     reweights examples: alpha upweights fraud samples,
                     gamma down-weights easy legitimate samples so training
                     focuses on hard/ambiguous transactions.
"""

import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 0.1,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority (fraud) class using SMOTE.

    sampling_strategy: desired minority/majority ratio after resampling.
      0.1 means 10 fraud samples per 100 legitimate — aggressive enough to
      help without flooding the training set with synthetic noise.

    Returns (X_resampled, y_resampled).
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    return smote.fit_resample(X, y)


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Compute XGBoost scale_pos_weight = n_negative / n_positive.

    This is used alongside SMOTE as a secondary signal, telling XGBoost's
    tree-building objective to treat each remaining fraud sample as if it
    represents more real-world fraud than its count suggests.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    return n_neg / max(n_pos, 1)
