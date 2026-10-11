import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ParameterSampler,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from prepare_data import (
    load_and_validate_data,
    get_features_and_target,
)
from preprocessing import build_preprocessor
from split_data import split_dataset, RANDOM_SEED


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    # 1. Reproduce the existing data partitions.
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    database_path = (PROJECT_ROOT / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")
    mlflow.set_experiment("AI4I-Predictive-Maintenance")

    # 2. Define the settings we want to explore.
    search_space = {
        "n_estimators": [100, 200],
        "max_depth": [8, 16, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", None],
        "class_weight": [None, "balanced"],
    }

    # Include the original random forest as a reference.
    baseline_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    }

    sampled_params = list(
        ParameterSampler(
            search_space,
            n_iter=8,
            random_state=RANDOM_SEED,
        )
    )

    # Avoid evaluating the reference twice if it was sampled.
    candidates = [baseline_params] + [
        params
        for params in sampled_params
        if params != baseline_params
    ]

    # 3. Use the same folds as our initial comparison.
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
    best_score = float("-inf")
    best_params = None
    best_run_id = None

    # 4. Create a parent run for the complete search.
    with mlflow.start_run(
        run_name="random_forest_tuning"
    ):
        mlflow.log_params(
            {
                "model_family": "random_forest",
                "sampled_configurations": 8,
                "evaluated_configurations": len(candidates),
                "cv_folds": 5,
                "random_seed": RANDOM_SEED,
                "training_rows": len(X_train),
                "training_failures": int(y_train.sum()),
            }
        )

        mlflow.set_tags(
            {
                "stage": "hyperparameter_tuning",
                "dataset": "AI4I 2020 synthetic",
                "evaluation": "training cross-validation",
                "selection_metric": "average_precision",
            }
        )

        # 5. Evaluate each configuration in a child run.
        for trial_number, params in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"\nTrial {trial_number}/{len(candidates)}",
                flush=True,
            )
            print(params, flush=True)

            estimator = RandomForestClassifier(
                **params,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )

            model_pipeline = Pipeline(
                steps=[
                    ("preprocessing", build_preprocessor()),
                    ("model", estimator),
                ]
            )

            with mlflow.start_run(
                run_name=f"rf_trial_{trial_number:02d}",
                nested=True,
            ) as trial_run:
                mlflow.log_params(estimator.get_params())

                mlflow.set_tag(
                    "configuration",
                    "reference"
                    if trial_number == 1
                    else "sampled",
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
                    scores = cv_results[
                        f"test_{metric_name}"
                    ]

                    metrics[
                        f"cv_{metric_name}_mean"
                    ] = float(scores.mean())

                    metrics[
                        f"cv_{metric_name}_std"
                    ] = float(scores.std())

                    for fold, score in enumerate(
                        scores,
                        start=1,
                    ):
                        mlflow.log_metric(
                            f"fold_{metric_name}",
                            float(score),
                            step=fold,
                        )

                mlflow.log_metrics(metrics)

                run_id = trial_run.info.run_id

                results.append(
                    {
                        "trial": trial_number,
                        "run_id": run_id,
                        **params,
                        **metrics,
                    }
                )

                score = metrics["cv_average_precision_mean"]

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_run_id = run_id

                print(f"Mean AP: {score:.4f}", flush=True)

        # 6. Record the winning configuration in the parent run.
        mlflow.log_metric(
            "best_cv_average_precision_mean",
            best_score,
        )

        mlflow.log_params(
            {
                f"best_{name}": value
                for name, value in best_params.items()
            }
        )

        mlflow.set_tag("best_trial_run_id", best_run_id)

        comparison = pd.DataFrame(results).sort_values(
            "cv_average_precision_mean",
            ascending=False,
        )

        comparison_path = OUTPUT_DIR / "rf_tuning_results.csv"
        comparison.to_csv(comparison_path, index=False)

        best_config = {
            "model_family": "random_forest",
            "parameters": {
                **best_params,
                "random_state": RANDOM_SEED,
                "n_jobs": 1,
            },
            "cv_average_precision_mean": best_score,
            "trial_run_id": best_run_id,
        }

        config_path = OUTPUT_DIR / "rf_best_config.json"
        config_path.write_text(
            json.dumps(best_config, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(config_path))

    print("\nBest configuration:")
    print(json.dumps(best_config, indent=2))

    print("\nTrial comparison:")
    print(
        comparison[
            [
                "trial",
                "cv_average_precision_mean",
                "cv_average_precision_std",
                "cv_roc_auc_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()