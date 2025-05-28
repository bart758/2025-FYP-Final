import pandas as pd
from collections.abc import Callable
from util.image import Image, importImages
from util.classifier_util import compare_classifiers, Classify
from util.extract_features import extractFeatures

def main(csv_path: str, save_path: str, features: list[Callable[..., float]], images_path: str = "./data", metadata_path: str = "./metadata.csv", hair_csv_path: str = './norm_region_hair_amount.csv', testing: bool = False):
    """Main function for image clasification.

    Imports features csv if it exists, if it does not exist uses images path 
    "./data" and metadata path "./metadata.csv" to create the cvs using
    ABCfeatures function.

    Separates the data into x and y parameters for use in classification,
    uses Logistic Regression to classify each image as melanoma or not melanoma
    and saves a csv with the results.

    Args:
        csv_path (str): path to the features csv with columns "patient_id" | "feat_A" | "feat_B" | ... | "feat_n" | "true_melanoma"
        save_path (str): save path for result csv with columns image_id | true_label | predicted_label | predicted_probability
        features (list[Callable[..., float]]): List of feature extraction functions, should be ordered as [feat_A, feat_B, ..., feat_n]
        images_path (str, optional): directory of images to be used in case the features csv does not exist. Defaults to "./data".
        metadata_path (str, optional): path to metadata.csv from original dataset. Defaults to "./metadata.csv".
        testing (bool, optional): displays performance of several classifiers over the data. Defaults to False.
    """

    # load dataset CSV file
    try:
        data_df = pd.read_csv(csv_path).dropna()
    except FileNotFoundError:
        images: list[Image] = importImages(images_path, metadata_path)
        data_df = extractFeatures(images, features, hair_csv_path)
        data_df.to_csv(csv_path, index=False)
        data_df = pd.read_csv(csv_path).dropna()

    # select only the baseline features
    baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    x_all = data_df[baseline_feats]
    y_all = data_df["true_melanoma"]

    if testing:
        compare_classifiers(x_all, y_all, n_iterations=30)
    else:
        Classify(x_all, y_all, save_path, data_df)


if __name__ == "__main__":
    
    features = [...]
    csv_path = "features_extended.csv"
    save_path = "result/result_extended.csv"
    hair_csv_path = "norm_region_hair_amount.csv"

    main(csv_path, save_path, features, hair_csv_path="norm_region_hair_amount.csv", testing=False)