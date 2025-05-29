import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)
import numpy as np
import pandas as pd
import cv2
import umap
from sklearn.model_selection import train_test_split
import os
from .img_util import get_hair_ratio
from .image import readImageFile
from .progressbar import progressbar


class ClassifierEvaluator:
    '''
    Utility class for evaluating binary classification models.

    Features:
        - Supports both "express" (text-based) and "visual" (plot-based) evaluation modes.
        -Supports both binary classification as well as well as multi class classification
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
    multiple (False: optional) a paremeter to state wherher the model evaluated classifies into multiple classes or not

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
    def __init__(self, clf, x_test, y_test, multiple = False):
        self.clf = clf
        self.x_test = x_test
        self.y_test = y_test
        self.avrage = "weighted" if multiple else "binary"
        self.roc_avrage = "weighted" if multiple else "macro"
        self.multi_class = "ovo" if multiple else "raise"
        self.y_pred = clf.predict(x_test)
        self.y_prob = clf.predict_proba(x_test)[:, 1] if not multiple else clf.predict_proba(x_test)
        self.metrics = {}

    def compute_metrics(self):
        self.metrics["accuracy"] = accuracy_score(self.y_test, self.y_pred)
        self.metrics["precision"] = precision_score(self.y_test, self.y_pred, zero_division=0, average=self.avrage)
        self.metrics["recall"] = recall_score(self.y_test, self.y_pred, zero_division=0, average=self.avrage)
        self.metrics["f1"] = f1_score(self.y_test, self.y_pred, zero_division=0, average=self.avrage)
        self.metrics["roc_auc"] = roc_auc_score(self.y_test, self.y_prob, average=self.roc_avrage, multi_class=self.multi_class)

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
        sns.heatmap(cm, annot=True, fmt='d', cmap=['#FFA500', '#fabe50', '#fcd48b', '#fce9c5', '#fff5e3'], xticklabels=["Non-Melanoma", "Melanoma"], yticklabels=["Non-Melanoma", "Melanoma"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        # plt.savefig('plots/confusion_balanced.pdf') # save figure separately

    def _plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {self.get_metrics()['roc_auc']:.2f}", c='#FFA500')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        # plt.savefig('plots/roc_final_final.pdf') # save figure separately

    def _plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, color='#FFA500')
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.tight_layout()

    def get_metrics(self):
        if not self.metrics:
            self.compute_metrics()
        return self.metrics

def EvaluateHairFeature(n_rus: int = 100, plotting: bool = True, plot_save_path: str = "plots", hair_annotations_path: str = "result/result.csv", 
                        directory: str = "data/Ol_data/", config: list[int] = [100, 220, 150, 10, 0.25, 0.15]) -> None:
    """Evaluates and visualizes the accuracy of the hair feature.

    Parameters
    ----------
    n_rus : int, optional
        Number of times that the accuracy will be calculated, by default 100
    plotting : bool, optional
        Enables plots, by default True
    plot_save_path : str, optional
        Save path for plot. Must be "save_path" withou slash at the end, by default "plots"
    hair_annotations_path : str, optional
        Path to manual hair annotations from the mandatory exercise, must contain: img_id | Rating_1, by default "result/result.csv"
    directory : str, optional
        Directory for images matching the dataset, by default "data/Ol_data/"
    config : list[int], optional
        Config for hair ratio calculation threshold, by default [100, 220, 150, 10, 0.25, 0.15]
    """

    class CFG:
        """Config file for Hair ratio calculation threshholds.
        """
        def __init__(self, x: list[int]):
            self.edge_low_threshold = x[0]
            self.edge_high_threshold = x[1]
            self.dark_spot_threshold = x[2]
            self.linelength_threshold = x[3]
            self.divergence_threshold = x[4]
            self.patchiness_threshold = x[5]
    
    cfg = CFG(config)

    def GetRatios(data_df: pd.DataFrame, images: list[tuple[str, np.ndarray]],  cfg: CFG) -> pd.DataFrame:
        """Get ratio for each image.

        Uses the get_har_ratio function to calculate the ratio of the hair coverage in the image.

        Parameters
        ----------
        data_df : pd.DataFrame
            Hair annotations from Mandatory assignment, columns: img_id | Rating_1 | ...
        images : list[tuple[str, np.ndarray]]
            List of img_id and corresponding image data
        cfg : CFG
            Cofig for hair ratio calculation

        Returns
        -------
        pd.DataFrame
            Same as input data_df + column Ratio
        """
        hair_df = pd.DataFrame()

        for img_id, image in progressbar(images, "Calculating ratios: "):
            hair_df.loc[img_id, 'Ratio'] = get_hair_ratio(image, cfg)
        
        hair_df.index.name = "img_id"
        hair_df = hair_df.merge(data_df, on="img_id", how="inner").drop(["Group_ID", "Unnamed: 7"], axis=1)

        return hair_df
    
    def CalculateAcuracy(hair_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Calculates accuracy of hair ratio calculation by comparing with manual annotions from mandatory exercise.

        Splits data into training and testin data. Gets threshold to split the data by using the train set and
        calculates the accuracy on test set.

        Parameters
        ----------
        hair_df : pd.DataFrame
            Must contain columns: img_id | Rating_1 | Ratio

        Returns
        -------
        float
            Accuracy calculated as #correctly_predicted_label/n_data
        pd.DataFrame
            Test data + columns: predicted_rating | correct prediction
        """
        
        X_train, X_test, y_train, y_test = train_test_split(hair_df["Ratio"], hair_df["Rating_1"])

        har_df_train = pd.merge(X_train, y_train, on="img_id", how="inner")
        hair_df_test = pd.merge(X_test, y_test, on="img_id", how="inner")

        hair_df_summary = har_df_train.groupby(by="Rating_1").describe()

        t_0 = hair_df_summary.loc[0.0, ("Ratio", "75%")]
        t_1 = hair_df_summary.loc[2.0, ("Ratio", "25%")]

        hair_df_test["predicted_rating"] = 2
        hair_df_test.loc[hair_df_test["Ratio"] <= t_0, "predicted_rating"] = 0
        hair_df_test.loc[(hair_df_test["Ratio"] >= t_0) & (hair_df_test["Ratio"] <= t_1), "predicted_rating"] = 1
        hair_df_test.loc[hair_df_test["Ratio"] >= t_1, "predicted_rating"] = 2

        hair_df_test["correct prediction"] = False
        hair_df_test["correct prediction"] = hair_df_test["Rating_1"] == hair_df_test["predicted_rating"]
        accuracy = sum(hair_df_test["correct prediction"])/len(hair_df_test["correct prediction"])
        return accuracy, hair_df_test
    
    def Plot(hair_df: pd.DataFrame, plot_save_path: str, accuracies: list[float]) -> None:
        """Plots accuracy information.

        Plots a boxplot of ratio distribution for each manual annotation lable for one accuracy run, 
        also plots the distribution of accuracy over n_runs.s

        Parameters
        ----------
        hair_df : pd.DataFrame
            Must contain columns: Rating_1 | Ratio
        plot_save_path : str
            Save path for plot. Must be "save_path" withou slash at the end.
        accuracies : list[float]
            List of accuracies over n_runs
        """
        plt.rcParams.update({'font.size': 15})

        zeros = hair_df[hair_df["Rating_1"] == 0]["Ratio"]
        ones = hair_df[hair_df["Rating_1"] == 1]["Ratio"]
        twos = hair_df[hair_df["Rating_1"] == 2]["Ratio"]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Hair ratio in test set by manual category")
        plt.xlabel("Manual hair category")
        plt.ylabel("Hair ratio")
        plt.text(0.6, 0.85, f"Accuracy: {accuracies[0]:.2f}%")
        plt.boxplot([zeros, ones, twos])
        plt.xticks(ticks=[1, 2, 3],labels=["No hair", "Some hair", "A lot of hair"])

        plt.subplot(1, 2, 2)
        plt.title(f"Accuracy distribuion over {len(accuracies)} runs")
        plt.boxplot(accuracies)
        plt.ylabel("Accuray in %")
        plt.xticks(ticks=[])

        plt.tight_layout()
        plt.savefig(f"{plot_save_path}/003-Hair_feat_eval.pdf" , bbox_inches='tight')
        plt.show()

    # Read dataset
    data_df = pd.read_csv(hair_annotations_path)
    data_df.set_index("img_id", inplace=True)

    # Import Images
    images: list[tuple[str, np.ndarray]] = list()
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    loaded: int = len(image_paths)

    for path in progressbar(image_paths, "Importing images:"):
        img_id = path.split("/")[-1]
        try:
            data_df.loc[img_id]
        except KeyError:
            loaded -= 1
            continue
        images.append((img_id, readImageFile(path)[0]))
    print(f"Loaded {loaded} images, contained in dataset.")

    # Run computation
    hair_df = GetRatios(data_df, images, cfg)
    accuracies: list[tuple[float, pd.DataFrame]] = list()
    for _ in progressbar(list(range(n_rus)), f"Running accuracy calculation: "):
        accuracies.append(CalculateAcuracy(hair_df))
    if plotting:
        Plot(accuracies[0][1], plot_save_path, [round(x[0]*100, 2) for x in accuracies])

def PlotProbability_vs_FeatureSpace():
    plt.rcParams.update({'font.size': 6*3})

    features = pd.read_csv("features_extended.csv")[["patient_id", "feat_A", "feat_B", "feat_C", "feat_D", "feat_E", "feat_F"]]
    pred_prob = pd.read_csv("result/result_extended.csv")[["patient_id", "predicted_probability"]]
    true_label = pd.read_csv("features_extended.csv")[["patient_id", "true_melanoma"]]

    data = features.merge(pred_prob, "inner", on="patient_id")
    data = data.merge(true_label, "inner", on="patient_id")

    data.set_index("patient_id", inplace=True)

    umap_model = umap.UMAP(n_neighbors=100, min_dist=0.1, n_components=1)
    X_umap = umap_model.fit_transform(data[["feat_A", "feat_B", "feat_C", "feat_D", "feat_E", "feat_F"]])

    data["umap"] = X_umap

    melanoma = data.loc[data["true_melanoma"] == True, :]
    non_melanoma = malanoma = data.loc[data["true_melanoma"] == False, :]

    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,2)
    plt.scatter(non_melanoma["umap"], non_melanoma["predicted_probability"], c="#333333", label="Non-melanoma")
    plt.scatter(melanoma["umap"], melanoma["predicted_probability"], c="#FFA500", label="Melanoma")
    plt.xlabel('Projected Feature Space')
    plt.yticks(ticks=[])
    plt.title('UMAP of Extended Model')

    features = pd.read_csv("features_baseline.csv")[["patient_id", "feat_A", "feat_B", "feat_C"]]
    pred_prob = pd.read_csv("result/result_baseline.csv")[["patient_id", "predicted_probability"]]
    true_label = pd.read_csv("features_baseline.csv")[["patient_id", "true_melanoma"]]

    data = features.merge(pred_prob, "inner", on="patient_id")
    data = data.merge(true_label, "inner", on="patient_id")

    data.set_index("patient_id", inplace=True)

    umap_model = umap.UMAP(n_neighbors=100, min_dist=0.1, n_components=1)
    X_umap = umap_model.fit_transform(data[["feat_A", "feat_B", "feat_C"]])

    data["umap"] = X_umap

    melanoma = data.loc[data["true_melanoma"] == True, :]
    non_melanoma = malanoma = data.loc[data["true_melanoma"] == False, :]

    plt.subplot(1,2,1)
    plt.scatter(non_melanoma["umap"], non_melanoma["predicted_probability"], c="#333333", label="Non-melanoma")
    plt.scatter(melanoma["umap"], melanoma["predicted_probability"], c="#FFA500", label="Melanoma")
    plt.xlabel('Projected Feature Space')
    plt.ylabel('Melanoma Probability')
    plt.title('UMAP of Baseline Model')
    plt.savefig("plots/007-UMAP-Baseline_and_extended-model.pdf", bbox_inches='tight')