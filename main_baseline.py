from collections.abc import Callable
from util.image import Image
from util.feature_A import asymmetry
from util.feature_B import compactness_score
from util.feature_C import get_multicolor_rate
from util.extract_features import ImportFeatures
from util.classifier import Classify

def main(csv_path: str, save_path: str, features: list[Callable[[Image], float]], images_path: str = "./data", metadata_path: str = "./metadata.csv", 
         multiple:bool = False, testing: bool = False, plots: bool= False):
    """Main function for image clasification.

    Imports features csv if it exists, if it does not exist uses images path 
    "./data" and metadata path "./metadata.csv" to create the cvs using
    ABCfeatures function.

    Separates the data into x and y parameters for use in classification,
    uses Logistic Regression to classify each image as melanoma or not melanoma
    and saves a csv with the results.

    Args:
        testing (bool, optional): displays performance of several classifiers over the data. Defaults to False.
    Parameters
    ----------
    csv_path : str
        Path to the features csv with columns "patient_id" | "feat_A" | "feat_B" | ... | "feat_n" | "true_melanoma"
    save_path : str
        Save path for result csv with columns image_id | true_label | predicted_label | predicted_probability
    features : list[Callable[[Image], float]]
        List of feature extraction functions, should be ordered as [feat_A, feat_B, ..., feat_n]
    images_path : str, optional
        Directory of images to be used in case the features csv does not exist, by default "./data"
    metadata_path : str, optional
        Path to metadata.csv from original dataset, by default "./metadata.csv"
    hair_csv_path : str, optional
        Path to csv of extracted hair ratios, by default './norm_region_hair_amount.csv'
    multiple : bool, optional
        True if multiple classification, by default False
    testing : bool, optional
        Run different splits of data through multiple classification algorithms and display plot of performance, by default False
    plots : bool, optional
        Performance plots for main classification, by default False
    """

    x_all, y_all, data_df = ImportFeatures(csv_path, images_path, metadata_path, features, multiple=multiple)

    Classify(x_all, y_all, save_path, data_df, multiple=multiple, plots = plots, testing=testing)


if __name__ == "__main__":
    
    features: list[Callable[[Image], float]] = [asymmetry, compactness_score, get_multicolor_rate]
    csv_path = "features_baseline.csv"
    save_path = "result/result_baseline.csv"
    hair_csv_path = "norm_region_hair_amount.csv"
    metadata_path = "dataset.csv"

    main(csv_path, save_path, features, metadata_path=metadata_path)