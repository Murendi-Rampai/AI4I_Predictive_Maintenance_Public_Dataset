from pathlib import Path

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from prepare_data import (
    load_and_validate_data,
    get_features_and_target,
)
from split_data import split_dataset, RANDOM_SEED
from preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    # 1. Load the data and reproduce our existing splits.
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

    # 2. Store experiment records in a local SQLite database.
    database_path = (
        PROJECT_ROOT / "mlflow.db"
    ).as_posix()

    mlflow.set_tracking_uri(
        f"sqlite:///{database_path}"
    )

    mlflow.set_experiment(
        "AI4I-Predictive-Maintenance"
    )

    # 3. Define the models we want to compare.
    models = {
        "dummy": DummyClassifier(
            strategy="prior",
        ),
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=RANDOM_SEED,
        ),
    }

    # 4. Define cross-validation within the training set.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
    }

    results = []

    # 5. Evaluate and record each model.
    for model_name, estimator in models.items():

        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    build_preprocessor(),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        with mlflow.start_run(
            run_name=model_name,
        ) as run:

            mlflow.log_params(
                {
                    "model_family": model_name,
                    "cv_folds": 5,
                    "random_seed": RANDOM_SEED,
                    "training_rows": len(X_train),
                    "training_failures": int(
                        y_train.sum()
                    ),
                }
            )

            # Record the estimator's actual settings.
            mlflow.log_params(
                estimator.get_params()
            )

            mlflow.set_tags(
                {
                    "stage": "baseline_comparison",
                    "dataset": "AI4I 2020 synthetic",
                    "evaluation": "training cross-validation",
                }
            )

            cv_results = cross_validate(
                estimator=model_pipeline,
                X=X_train,
                y=y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )

            metrics = {}

            for metric_name in scoring:
                fold_scores = cv_results[
                    f"test_{metric_name}"
                ]

                metrics[
                    f"cv_{metric_name}_mean"
                ] = float(fold_scores.mean())

                metrics[
                    f"cv_{metric_name}_std"
                ] = float(fold_scores.std())

                # Keep the individual fold results too.
                for fold, score in enumerate(
                    fold_scores,
                    start=1,
                ):
                    mlflow.log_metric(
                        f"fold_{metric_name}",
                        float(score),
                        step=fold,
                    )

            mlflow.log_metrics(metrics)

            results.append(
                {
                    "model": model_name,
                    "run_id": run.info.run_id,
                    **metrics,
                }
            )

            print(f"\nModel: {model_name}")

            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")

    # 6. Save a comparison table.
    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "cv_average_precision_mean",
        ascending=False,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_DIR / "baseline_comparison.csv",
        index=False,
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()