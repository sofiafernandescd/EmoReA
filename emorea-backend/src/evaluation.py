import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix with percentages as colors 
    and absolute counts shown in parentheses.

    Args:
        y_true (list/array): Ground truth labels.
        y_pred (list/array): Predicted labels.
        labels (list): List of class labels (in order). If None, uses unique labels from y_true.
        title (str): Title of the plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if labels is None:
        labels = np.unique(y_true)

    # Normalize row-wise to percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent)  # handle divide by zero

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)

    # Title + colorbar
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Percentage")

    # Axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text in cells: show percent + count
    thresh = cm_percent.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_percent[i, j]*100:.1f}%\n({cm[i, j]})"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_percent[i, j] > thresh else "black"
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.show()