from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from data_load import load_dataset
from preprocessing import preprocess_dataset
from epochs import create_epochs
from features import extract_epoch_features
from split import make_group_shuffle_split
from pipeline import get_models
from visual import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

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
    X_train, X_test, y_train, y_test, groups_train, groups_test = make_group_shuffle_split(
        X=X_features,
        y=y_epochs,
        groups=groups_epochs,
        test_size=0.2,
        random_state=42,
    )

    print("Train subjects:", len(set(groups_train)))
    print("Test subjects:", len(set(groups_test)))

    #  Cargar de modelos y resultados
    models = get_models()
    results = []

    # Carpeta para guardar las figuras generadas durante la evaluación
    figures_dir = OUTPUT_DIR / "Figuras"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Entrenamiento y evaluación de cada modelo
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Se calculan métricas principales para comparar modelos
        results.append({
            "Modelo": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-score": f1_score(y_test, y_pred, average="weighted"),
        })

        print(f"\n{model_name}")
        print(classification_report(y_test, y_pred))

        # Crear carpetas
        model_figures_dir = figures_dir / model_name
        model_figures_dir.mkdir(parents=True, exist_ok=True)

        # Generar imagenes
        plot_confusion_matrix(
            model,
            X_test,
            y_test,
            save_path=model_figures_dir / "confusion_matrix.png",
        )

        plot_roc_curve(
            model,
            X_test,
            y_test,
            save_path=model_figures_dir / "roc_curve.png",
        )

        plot_precision_recall_curve(
            model,
            X_test,
            y_test,
            save_path=model_figures_dir / "precision_recall_curve.png",
        )

    # Construir dataframa con los resultados obtenidos en los modelos
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

    print("\nResultados finales:")
    print(results_df)


if __name__ == "__main__":
    main()