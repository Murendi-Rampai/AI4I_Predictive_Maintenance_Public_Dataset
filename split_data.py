from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from prepare_data import (
    load_and_validate_data,
    get_features_and_target,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_SEED = 42


def split_dataset(X, y):
    """Create stratified 60/20/20 training, validation, and test sets."""
    #stratify asks the split to preserve approximately the same proportion of failure and nonfailure labels in each set.
    # First reserve 20% of the complete dataset for testing.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Reserve 25% of the remaining 80% for validation.
    # 25% × 80% = 20% of the complete dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df, disagreements = load_and_validate_data()
    X, y = get_features_and_target(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(X, y)

    # Check that no row appears in more than one set.
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    # Check that every original row belongs to a set.
    assert train_indices | val_indices | test_indices == set(X.index)

    summary = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(target),
                "failures": int(target.sum()),
                "failure_percent": round(target.mean() * 100, 2),
            }
            for name, target in [
                ("training", y_train),
                ("validation", y_val),
                ("test", y_test),
            ]
        ]
    )

    print(summary.to_string(index=False))
    print("\nChecks passed: no overlapping rows; all rows accounted for.")

    # Save which original rows belong to each set.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_manifest = pd.concat(
        [
            pd.DataFrame(
                {
                    "row_index": features.index,
                    "split": name,
                }
            )
            for name, features in [
                ("training", X_train),
                ("validation", X_val),
                ("test", X_test),
            ]
        ],
        ignore_index=True,
    )

    manifest_path = OUTPUT_DIR / "split_manifest.csv"
    split_manifest.to_csv(manifest_path, index=False)

    print(f"\nSaved split manifest to: {manifest_path}")