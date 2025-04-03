import sys
from os.path import join

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from util.img_util import readImageFile, saveImageFile
from util.inpaint_util import removeHair


def main(csv_path, save_path):
    # load dataset CSV file
    data_df = pd.read_csv(csv_path)

    # select only the baseline features.
    baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    x_all = data_df[baseline_feats]
    y_all = data_df["label"]

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
    result_df = data_df.loc[x_test.index, ["filename"]].copy()
    result_df['true_label'] = y_test.values
    result_df['predicted_label'] = y_pred
    result_df.to_csv(save_path, index=False)
    print("Results saved to:", save_path)


if __name__ == "__main__":
    csv_path = "./dataset.csv"
    save_path = "./result/result_baseline.csv"

    main(csv_path, save_path)
