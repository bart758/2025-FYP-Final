import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)


class ClassifierEvaluator:
    '''
    Utility class for evaluating binary classification models.

    Features:
        - Supports both "express" (text-based) and "visual" (plot-based) evaluation modes.
        - Computes standard performance metrics: accuracy, precision, recall, F1-score, and ROC AUC.
        - Generates classification report and confusion matrix.
        - In "visual" mode, plots:
        - Confusion matrix heatmap
        - ROC curve with AUC
        - Precision-Recall curve

    Parameters
    ----------
    clf
        The trained classifier.
    x_test
        The test data.
    y_test
        Correct labels of the test data.

    Returns
    -------
    Any
        The metrics of your model (either in text form or plots).

    Usage
    -----
        evaluator = ClassifierEvaluator(clf, x_test, y_test)
        evaluator.evaluate(mode="express")  | Prints metrics to terminal
        evaluator.evaluate(mode="visual")   | Displays evaluation plots
    '''
    def __init__(self, clf, x_test, y_test):
        self.clf = clf
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = clf.predict(x_test)
        self.y_prob = clf.predict_proba(x_test)[:, 1]
        self.metrics = {}

    def compute_metrics(self):
        self.metrics["accuracy"] = accuracy_score(self.y_test, self.y_pred)
        self.metrics["precision"] = precision_score(self.y_test, self.y_pred, zero_division=0)
        self.metrics["recall"] = recall_score(self.y_test, self.y_pred, zero_division=0)
        self.metrics["f1"] = f1_score(self.y_test, self.y_pred, zero_division=0)
        self.metrics["roc_auc"] = roc_auc_score(self.y_test, self.y_prob)

    def express(self):
        self.compute_metrics()
        print("\n--- CLASSIFIER REPORT ---")
        for k, v in self.metrics.items():
            print(f"{k.capitalize():<12}: {v:.4f}")
        print("\nConfusion Matrix:\n", confusion_matrix(self.y_test, self.y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, self.y_pred, digits=4))

    def visual(self):
        self._plot_confusion_matrix()
        self._plot_roc_curve()
        self._plot_precision_recall_curve()
        plt.show()

    def _plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Non-Melanoma", "Melanoma"], yticklabels=["Non-Melanoma", "Melanoma"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

    def _plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {self.get_metrics()['roc_auc']:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()

    def _plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, color="blue")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.tight_layout()

    def get_metrics(self):
        if not self.metrics:
            self.compute_metrics()
        return self.metrics
