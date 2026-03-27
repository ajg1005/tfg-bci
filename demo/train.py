from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from data_load import load_dataset
from preprocessing import preprocess_dataset
from epochs import create_epochs
from features import extract_epoch_features
from split import make_group_shuffle_split,make_group_kfold_splits
from pipeline import get_models
from visual import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from sklearn.base import clone

# Rutas para el dataset y las figuras
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "adhdata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent 


def main():
    # Cargar el dataset,limpiar y preprocesar
    df = load_dataset(CSV_PATH)
    df_clean, eeg_cols = preprocess_dataset(df)

    # Segmentar en epochs
    X_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=128,
        step_size=64,
    )

    print("Shape X_epochs:", X_epochs.shape)
    print("Shape y_epochs:", y_epochs.shape)
    print("Shape groups_epochs:", groups_epochs.shape)

    # Extraer caracteristicas
    X_features = extract_epoch_features(X_epochs, eeg_cols)

    print("Shape X_features:", X_features.shape)

    # Dividir en entrenamiento y prueba por sujetos
    cv_splits = make_group_kfold_splits(
    X_features,
    y_epochs,
    groups_epochs,
    n_splits=5,
    )
 
    # Cargar modelos
    models = get_models()

    # Lista para guardar métricas de todos los folds y modelos
    results = []

    # Guardar información necesaria para luego generar imágenes solo del mejor fold
    fold_storage = []

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


        # Entrenamiento y evaluación de cada modelo
        for model_name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            results.append({
            "Modelo": model_name,
            "Fold": fold,
            "Accuracy": acc,
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            })

            if hasattr(fitted_model, "predict_proba"):
                y_score = fitted_model.predict_proba(X_test)[:, 1]
            elif hasattr(fitted_model, "decision_function"):
                y_score = fitted_model.decision_function(X_test)
            else:
                y_score = None

            fold_storage.append({
            "Modelo": model_name,
            "Fold": fold,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_score": y_score,
            "Accuracy": acc,
            })

            print(f"\n{model_name} - Fold {fold}")
            print(classification_report(y_test, y_pred, zero_division=0))

    # Construir dataframa con los resultados obtenidos en los modelos
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

    print("\nResultados finales:")
    print(results_df)
    summary_df = results_df.groupby("Modelo").agg({
    "Accuracy": ["mean", "std"],
    "Precision": ["mean", "std"],
    "Recall": ["mean", "std"],
    "F1-score": ["mean", "std"],
    }).round(4)

    # Elegir el mejor fold de cada modelo según Accuracy
    best_folds_df = results_df.loc[results_df.groupby("Modelo")["Accuracy"].idxmax()]

    print("\nMejor fold de cada modelo:")
    print(best_folds_df[["Modelo", "Fold", "Accuracy", "F1-score"]])

     # Generar imágenes solo para el mejor fold de cada modelo
    for _, row in best_folds_df.iterrows():
        model_name = row["Modelo"]
        best_fold = row["Fold"]

        selected = next(
            item for item in fold_storage
            if item["Modelo"] == model_name and item["Fold"] == best_fold
        )

        y_test = selected["y_test"]
        y_pred = selected["y_pred"]
        y_score = selected["y_score"]

        model_figures_dir = figures_dir / model_name / f"best_fold_{best_fold}"
        model_figures_dir.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(
            y_test,
            y_pred,
            save_path=model_figures_dir / "confusion_matrix.png",
        )

        
        plot_roc_curve(
            y_test,
            y_score,
            save_path=model_figures_dir / "roc_curve.png",
        )

        plot_precision_recall_curve(
            y_test,
            y_score,
            save_path=model_figures_dir / "precision_recall_curve.png",
        )

    print("\nResumen final de cross-validation:")
    print(summary_df)


if __name__ == "__main__":
    main()