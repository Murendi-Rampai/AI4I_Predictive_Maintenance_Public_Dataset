from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

# The only columns the model will receive.
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"

FAILURE_MODE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_and_validate_data():
    """Load AI4I and check its structure and labels."""
    df = pd.read_csv(DATA_PATH)

    # Check that all required columns exist.
    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
        + FAILURE_MODE_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Check that outcome labels contain only 0 or 1.
    label_columns = [TARGET_COLUMN] + FAILURE_MODE_COLUMNS

    if not df[label_columns].isin([0, 1]).all().all():
        raise ValueError(
            "Failure labels must contain only 0 or 1."
        )

    # Check that product types are recognised.
    if not df["Type"].isin(["L", "M", "H"]).all():
        raise ValueError(
            "Product type must be L, M, or H."
        )

    # Audit whether the overall label agrees with the mode flags.
    any_failure_mode = df[FAILURE_MODE_COLUMNS].max(axis=1)

    disagreements = df[
        df[TARGET_COLUMN] != any_failure_mode
    ].copy()

    return df, disagreements


def get_features_and_target(df):
    """Separate the allowed model inputs from the outcome."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    return X, y


if __name__ == "__main__":
    df, disagreements = load_and_validate_data()
    X, y = get_features_and_target(df)

    print("Validation passed.")

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nFeature shape:", X.shape)
    print("Target shape:", y.shape)

    print(
        "\nOverall label / failure-mode disagreements:",
        len(disagreements),
    )

    print("\nFirst five input rows:")
    print(X.head())

    print("\nFirst five target values:")
    print(y.head())