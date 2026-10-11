import pandas as pd

from prepare_data import (
    load_and_validate_data,
    get_features_and_target,
)
from split_data import split_dataset
from preprocessing import build_preprocessor


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

preprocessor = build_preprocessor()

# Learn preprocessing settings from training data only.
X_train_prepared = preprocessor.fit_transform(X_train)

# Apply those same settings to validation data.
X_val_prepared = preprocessor.transform(X_val)

feature_names = (
    preprocessor.named_steps["columns"]
    .get_feature_names_out()
)

prepared_preview = pd.DataFrame(
    X_train_prepared,
    columns=feature_names,
    index=X_train.index,
)

print("Original training shape:", X_train.shape)
print("Prepared training shape:", X_train_prepared.shape)
print("Prepared validation shape:", X_val_prepared.shape)

print("\nPrepared feature names:")
print(feature_names.tolist())

print("\nFirst five prepared training rows:")
print(prepared_preview.head().to_string())

print("\nMissing values after preprocessing:")
print(prepared_preview.isna().sum().to_string())