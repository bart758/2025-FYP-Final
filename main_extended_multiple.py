import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections.abc import Callable

from util.progressbar import progressbar
from util.image import Image, importImages
from util.Feature_A import asymmetry
from util.Feature_B import compactness_score
from util.Feature_C import get_multicolor_rate
from util.Feature_D import find_max_diameter
from util.Feature_E import is_growing
from util.evaluator_util import ClassifierEvaluator
from util.optimal_classifier_util import compare_classifiers
from util.hair_feature_util import hair_import, hair_ratio

def extractFeatures(images: list[Image], extraction_functions: list[Callable[..., float]]) -> pd.DataFrame:
    """Extracts features from list of images using funtions from extraction_functions list and saves them into a dataframe.

    Args:
        images (list[Image]): list of Image objects
        extraction_functions (list[Callable[..., float]]): List of feature extraction functions, should be ordered as [feat_A, feat_B, ..., feat_n]

    Returns:
        pd.DataFrame: Columns "patient_id" | "feat_A" | "feat_B" | ... | "feat_n" | "true_melanoma"
    """

    try:
        hair_df = pd.read_csv(hair_csv_path).dropna()
        hair_df.set_index('ImageID', inplace=True)
    except FileNotFoundError:
        hair_df = hair_import(images, hair_csv_path)

    features_df = pd.DataFrame(columns=["patient_id", "true_melanoma"])
    counter = 0 

    for i_image, image in progressbar(list(enumerate(images)), "Proccesing features: "):
        features_df.loc[i_image, "patient_id"] = image

        for i_func, func in enumerate(extraction_functions):
            variables = list()
            arg_count = func.__code__.co_argcount
            args = func.__code__.co_varnames[0: arg_count]

            for arg in args:
                if arg == "image":
                    variables.append(image)
                if arg == "hair_df":
                    variables.append(hair_df)

            try:
                features_df.loc[i_image, f"feat_{chr(i_func+65)}"] = func(*tuple(variables))
            except (FileNotFoundError, ValueError) as e: # if mask does not exist in masks folder
                counter += 1
                print(e)

        features_df.loc[i_image, "diagnostic"] = image.metadata["diagnostic"]
        features_df.loc[i_image, "true_melanoma"] = True if image.metadata["diagnostic"] == "MEL" else False

    print(f"There was an error proccessing {counter} images.")
    return features_df
            

def main(csv_path: str, save_path: str, features: list[Callable[..., float]], images_path: str = "./data", metadata_path: str = "./metadata.csv", hair_csv_path: str = './norm_region_hair_amount.csv', testing: bool = False):

    # load dataset CSV file
    try:
        data_df = pd.read_csv(csv_path).dropna()
    except FileNotFoundError:
        images: list[Image] = importImages(images_path, metadata_path)
        data_df = extractFeatures(images, features)
        data_df.to_csv(csv_path, index=False)
        data_df = pd.read_csv(csv_path).dropna()

    # select only the baseline features
    baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    x_all = data_df[baseline_feats]
    y_all = data_df["diagnostic"]

    if testing:
        compare_classifiers(x_all, y_all, n_iterations=30)
    else:
        # split the dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

        # train the classifier

        clf = LogisticRegression(max_iter=2000, verbose=0, class_weight='balanced', solver='liblinear', penalty='l1')
        clf.fit(x_train, y_train)

        # test the trained classifier
        probs = clf.predict_proba(x_test)[:, 1]

        y_pred = clf.predict(x_test)

        # evaluate the classifier
        evaluator = ClassifierEvaluator(clf, x_test, y_test, multiple=True)
        evaluator.express()

        # write test results to a CSV file
        result_df = data_df.loc[x_test.index, ["patient_id"]].copy()
        result_df['true_label'] = y_test.values
        result_df['predicted_label'] = str(y_pred[0])
        result_df['predicted_probability'] = str(probs[0])
        result_df.to_csv(save_path, index=False)
        print("Results saved to:", save_path)


if __name__ == "__main__":
    
    features = [asymmetry, compactness_score, get_multicolor_rate, find_max_diameter, is_growing, hair_ratio]
    csv_path = "features_extended.csv"
    save_path = "result/result_extended_multi.csv"
    hair_csv_path = "norm_region_hair_amount.csv"

    main(csv_path, save_path, features, hair_csv_path="norm_region_hair_amount.csv", testing=False)

