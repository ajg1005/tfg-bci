from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

# Crear figuras y guardarlas
def _save_fig(fig, save_path):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_confusion_matrix(y_test, y_score, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_score, ax=ax)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_roc_curve(y_test, y_score, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_precision_recall_curve(y_test, y_score, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=ax)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_model_metric_bar(summary_df, metric_name, save_path=None):
    plot_df = summary_df.copy()
    means = plot_df[(metric_name, "mean")]
    stds = plot_df[(metric_name, "std")]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(means.index, means.values, yerr=stds.values, capsize=4)
    ax.set_title(f"Comparación de modelos - {metric_name}")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Modelo")
    plt.xticks(rotation=20)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    plt.close(fig)