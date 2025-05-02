import sys
from os.path import join

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from util.img_util import saveImageFile, importImages
from util.inpaint_util import removeHair
from util.progressbar import progressbar
from util.image import Image

from util.Feature_A import get_asymmetry
from util.Feature_B import convexity_score
from util.Feature_C import get_multicolor_rate

def ABCfeatures(images: list[Image]) -> pd.DataFrame:
    features_df = pd.DataFrame(columns=["patient_id", "feat_A", "feat_B", "feat_C", "true_melanoma"])

    for i, image in enumerate(progressbar(sorted(images))):
        features_df.loc[i, "patient_id"] = image
        try:
            features_df.loc[i, "feat_A"] = get_asymmetry(image.mask)
            features_df.loc[i, "feat_B"] = convexity_score(image.mask)
            features_df.loc[i, "feat_C"] = get_multicolor_rate(image.color, image.mask, 15)
        except:
            pass
        features_df.loc[i, "true_melanoma"] = True if image.metadata["diagnostic"] == "MEL" else False

    return features_df.dropna()

def main(csv_path, save_path):
    # load dataset CSV file
    try:
        data_df = pd.read_csv(csv_path).dropna()
    except FileNotFoundError:
        images = importImages(images_path, metadata_path)
        data_df = ABCfeatures(images)

    # select only the baseline features.
    baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    x_all = data_df[baseline_feats]
    y_all = data_df["true_melanoma"]

    # split the dataset into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)

    # train the classifier (using logistic regression as an example)
    clf = LogisticRegression(max_iter=1000, verbose=1)
    clf.fit(x_train, y_train)

    # test the trained classifier
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

    # write test results to CSV.
    result_df = data_df.loc[x_test.index, ["patient_id"]].copy()
    result_df['true_label'] = y_test.values
    result_df['predicted_label'] = y_pred
    result_df.to_csv(save_path, index=False)
    print("Results saved to:", save_path)


if __name__ == "__main__":
    metadata_path = "./metadata.csv"
    images_path = "./data"
    csv_path = "./features.csv"
    save_path = "./result/result_baseline.csv"

    main(csv_path, save_path)
