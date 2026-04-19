from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from data_load import load_dataset
from epochs import create_epochs
from features import extract_epoch_features
from pipeline import get_models
from preprocessing import preprocess_dataset
from spectral_features import extract_spectral_features
from split import make_group_kfold_splits
from visual import (
    plot_confusion_matrix,
    plot_model_metric_bar,
    plot_roc_curve,
)

# Rutas para el dataset
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "adhdata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent


def main():
    # Cargar el dataset, limpiar y preprocesar
    df = load_dataset(CSV_PATH)
    df_clean, eeg_cols = preprocess_dataset(df)

    # Segmentar en epochs
    X_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=1920,
        step_size=960,
    )

    print("Shape X_epochs:", X_epochs.shape)
    print("Shape y_epochs:", y_epochs.shape)
    print("Shape groups_epochs:", groups_epochs.shape)

    # Features temporales
    X_time = extract_epoch_features(X_epochs, eeg_cols)

    # Features espectrales
    X_spectral = extract_spectral_features(
        X_epochs=X_epochs,
        channel_names=eeg_cols,
        sfreq=128,
        nperseg=1920,
    )

    # Features combinadas
    X_combined = pd.concat(
        [X_time.reset_index(drop=True), X_spectral.reset_index(drop=True)],
        axis=1,
    )

    X_features = X_combined

    print("Shape X_features:", X_features.shape)

    # CV cross-subject sobre todo el dataset
    cv_splits = make_group_kfold_splits(
        X_features,
        y_epochs,
        groups_epochs,
        n_splits=5,
    )

    # Cargar modelos
    models = get_models()

    # Guardar métricas de todos los folds y modelos
    results = []

    # Guardar predicciones out-of-fold para figuras finales
    oof_predictions = {
        model_name: {"y_true": [], "y_pred": [], "y_score": []}
        for model_name in models.keys()
    }

    figures_dir = OUTPUT_DIR / "Figuras"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for split_data in cv_splits:
        fold = split_data["fold"]
        X_train = split_data["X_train"]
        X_test = split_data["X_test"]
        y_train = split_data["y_train"]
        y_test = split_data["y_test"]
        groups_train = split_data["groups_train"]
        groups_test = split_data["groups_test"]

        print(f"\nFold {fold}")
        print("Sujetos train:", len(set(groups_train)))
        print("Sujetos test:", len(set(groups_test)))

        overlap_fold = set(groups_train) & set(groups_test)
        print("Solapamiento train/test en fold:", len(overlap_fold))

        # Entrenamiento y evaluación de cada modelo
        for model_name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)

            # Scores para ROC si el modelo los soporta
            if hasattr(fitted_model, "predict_proba"):
                y_score = fitted_model.predict_proba(X_test)[:, 1]
            elif hasattr(fitted_model, "decision_function"):
                y_score = fitted_model.decision_function(X_test)
            else:
                y_score = None

            # Guardar predicciones OOF
            oof_predictions[model_name]["y_true"].extend(y_test)
            oof_predictions[model_name]["y_pred"].extend(y_pred)
            if y_score is not None:
                oof_predictions[model_name]["y_score"].extend(y_score)

            # Métricas por epoch
            acc_epoch = accuracy_score(y_test, y_pred)
            bal_acc_epoch = balanced_accuracy_score(y_test, y_pred)
            prec_epoch = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            rec_epoch = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            f1_epoch = f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            )

            results.append({
                "Modelo": model_name,
                "Fold": fold,
                "Accuracy_epoch": acc_epoch,
                "BalancedAccuracy_epoch": bal_acc_epoch,
                "Precision_epoch": prec_epoch,
                "Recall_epoch": rec_epoch,
                "F1_epoch": f1_epoch,
            })

            print(
                f"{model_name} - Fold {fold} | "
                f"Accuracy epoch: {acc_epoch:.4f} | "
                f"Balanced Acc: {bal_acc_epoch:.4f} | "
                f"F1 epoch: {f1_epoch:.4f}"
            )

    # Construir dataframe con los resultados
    results_df = pd.DataFrame(results)

    summary_df = results_df.groupby("Modelo").agg({
        "Accuracy_epoch": ["mean", "std"],
        "BalancedAccuracy_epoch": ["mean", "std"],
        "Precision_epoch": ["mean", "std"],
        "Recall_epoch": ["mean", "std"],
        "F1_epoch": ["mean", "std"],
    }).round(4)

    print("\nRESUMEN CV CROSS-SUBJECT")
    print(summary_df)

    # Guardar resultados
    results_df.to_csv(OUTPUT_DIR / "cv_results_by_fold.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "cv_results_summary.csv")

    # Elegir mejor modelo usando la media de CV
    best_model_name = summary_df[("F1_epoch", "mean")].idxmax()
    print(f"\nMejor modelo según media de F1 por epoch en CV: {best_model_name}")

    # Figura 1: comparación de modelos por Balanced Accuracy
    plot_model_metric_bar(
        summary_df,
        metric_name="BalancedAccuracy_epoch",
        save_path=figures_dir / "cv_model_comparison_balanced_accuracy.png",
    )

    # Figura 2: comparación de modelos por F1
    plot_model_metric_bar(
        summary_df,
        metric_name="F1_epoch",
        save_path=figures_dir / "cv_model_comparison_f1.png",
    )

    # Figura 3: matriz de confusión global out-of-fold del mejor modelo
    best_y_true = oof_predictions[best_model_name]["y_true"]
    best_y_pred = oof_predictions[best_model_name]["y_pred"]

    plot_confusion_matrix(
        best_y_true,
        best_y_pred,
        save_path=figures_dir / f"{best_model_name}_cv_confusion_matrix.png",
    )

    # Figura 4: curva ROC global out-of-fold del mejor modelo
    best_y_score = oof_predictions[best_model_name]["y_score"]

    if len(best_y_score) == len(best_y_true) and len(best_y_score) > 0:
        plot_roc_curve(
            best_y_true,
            best_y_score,
            save_path=figures_dir / f"{best_model_name}_cv_roc_curve.png",
        )
    else:
        print(f"\nEl modelo {best_model_name} no dispone de scores continuos para ROC.")

    # Entrenar mejor modelo con todos los datos para despliegue/demo
    best_model = clone(models[best_model_name])
    best_model.fit(X_features, y_epochs)
    print("\nModelo final entrenado con todos los datos para despliegue/demo.")


if __name__ == "__main__":
    main()