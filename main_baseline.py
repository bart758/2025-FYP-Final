# def ABCfeatures(images: list[Image]) -> pd.DataFrame:
#     """Extracts features from list of images and saves into dataframe.

#     Args:
#         images (list[Image]): list of Image objects

#     Returns:
#         pd.DataFrame: Columns "patient_id" | "feat_A" | "feat_B" | "feat_C" | "true_melanoma"
#     """
#     features_df = pd.DataFrame(columns=["patient_id", "feat_A", "feat_B", "feat_C", "true_melanoma"])

#     for i, image in enumerate(progressbar(sorted(images))):
#         features_df.loc[i, "patient_id"] = image
#         try:
#             features_df.loc[i, "feat_A"] = get_asymmetry(image.mask)
#             features_df.loc[i, "feat_B"] = convexity_score(image.mask)
#             features_df.loc[i, "feat_C"] = get_multicolor_rate(image.color, image.mask, 15)
#         except:
#             pass
#         features_df.loc[i, "true_melanoma"] = True if image.metadata["diagnostic"] == "MEL" else False

#     return features_df.dropna()

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections.abc import Callable

from util.progressbar import progressbar
from util.image import Image, importImages
from util.Feature_A import asymmetry
from util.Feature_B import compactness_score
from util.Feature_C import get_multicolor_rate

def extractFeatures(images: list[Image], extraction_functions: list[Callable[..., float]]) -> pd.DataFrame:
    """Extracts features from list of images using funtions from extraction_functions list and saves them into a dataframe.

    Args:
        images (list[Image]): list of Image objects
        extraction_functions (list[Callable[..., float]]): List of feature extraction functions, should be ordered as [feat_A, feat_B, ..., feat_n]

    Returns:
        pd.DataFrame: Columns "patient_id" | "feat_A" | "feat_B" | ... | "feat_n" | "true_melanoma"
    """
    features_df = pd.DataFrame(columns=["patient_id", "feat_A", "feat_B", "feat_C", "true_melanoma"])

    for i_image, image in enumerate(images):
        features_df.loc[i_image, "patient_id"] = image
        features_df.loc[i_image, "true_melanoma"] = True if image.metadata["diagnostic"] == "MEL" else False

        for i_func, func in enumerate(extraction_functions):
            variables = list()
            arg_count = func.__code__.co_argcount
            args = func.__code__.co_varnames[0: arg_count]

            for arg in args:
                if arg == "image":
                    variables.append(image.color)
                try:
                    if arg == "mask":
                        variables.append(image.mask)
                except FileNotFoundError as e:
                    variables = []
                    break

            if variables:
                features_df.loc[i_image, f"feat_{chr(i_func+65)}"] = func(*tuple(variables))

    return features_df
            

def main(csv_path: str, save_path: str, images_path: str = "./data", metadata_path: str = "./metadata.csv"):
    """Main function for image clasification.

    Imports features csv if it exists, if it does not exist uses images path 
    "./data" and metadata path "./metadata.csv" to create the cvs using
    ABCfeatures function.

    Separates the data into x and y parameters for use in classification,
    uses Logistic Regression to classify each image as melanoma or not melanoma
    and saves a csv with the results.

    Args:
        csv_path (str): path to the features csv with columns "patient_id" | "feat_A" | "feat_B" | ... | "feat_n" | "true_melanoma"
        save_path (str): save path for result csv with columns image_id | true_label | predicted_label
        images_path (str, optional): directory of images to be used in case the features csv does not exist. Defaults to "./data".
        metadata_path (str, optional): path to metadata.csv from original dataset. Defaults to "./metadata.csv".
    """
    # load dataset CSV file
    try:
        data_df = pd.read_csv(csv_path).dropna()
    except FileNotFoundError:
        images: list[Image] = importImages(images_path, metadata_path)
        data_df = extractFeatures(images, [asymmetry, compactness_score, get_multicolor_rate])
        data_df.to_csv("features.csv")
        data_df = pd.read_csv(csv_path).dropna()

    print(data_df.head())

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
    csv_path = "features.csv"
    save_path = "result/result_baseline.csv"

    main(csv_path, save_path)

