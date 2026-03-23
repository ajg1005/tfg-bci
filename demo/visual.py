from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

# Crear figuras y guardarlas
def _save_fig(fig, save_path):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_confusion_matrix(model, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_precision_recall_curve(model, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    plt.close(fig)