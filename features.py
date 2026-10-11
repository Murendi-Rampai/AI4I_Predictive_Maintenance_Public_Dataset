import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from prepare_data import FEATURE_COLUMNS


NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temperature_delta_k",
    "power_w",
    "wear_torque_min_nm",
]

CATEGORICAL_COLUMNS = ["Type"]


class EngineeringFeatures(TransformerMixin, BaseEstimator):
    """Add engineering features using only allowed model inputs."""

    def fit(self, X, y=None):
        # These calculations do not learn anything from the data.
        return self

    def transform(self, X):
        # Select allowed inputs and avoid changing the original table.
        result = X[FEATURE_COLUMNS].copy()

        result["temperature_delta_k"] = (
            result["Process temperature [K]"]
            - result["Air temperature [K]"]
        )

        result["power_w"] = (
            result["Torque [Nm]"]
            * result["Rotational speed [rpm]"]
            * (2 * np.pi / 60)
        )

        result["wear_torque_min_nm"] = (
            result["Tool wear [min]"]
            * result["Torque [Nm]"]
        )

        return result