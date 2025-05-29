import pandas as pd
from collections.abc import Callable
from util.progressbar import progressbar
from util.hair_feature_util import hair_import
from .feature_F import hair_ratio
from .image import Image, importImages

def extractFeatures(images: list[Image], extraction_functions: list[Callable[..., float]], hair_csv_path: str | None = None) -> pd.DataFrame:
    """Extracts features from list of images using funtions from extraction_functions list and saves them into a dataframe.

    Args:
        images (list[Image]): list of Image objects
        extraction_functions (list[Callable[..., float]]): List of feature extraction functions, should be ordered as [feat_A, feat_B, ..., feat_n]

    Returns:
        pd.DataFrame: Columns "patient_id" | "feat_A" | "feat_B" | ... | "feat_n" | "true_melanoma"
    """
    if hair_ratio in extraction_functions:
        try:
            if hair_csv_path is not None:
                hair_df = pd.read_csv(hair_csv_path).dropna()
                hair_df.set_index('ImageID', inplace=True)
            else:
                raise FileNotFoundError
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

def ImportFeatures(csv_path: str, images_path: str, metadata_path: str, features: list[Callable[[Image], float]], masks_path: str = "masks/",
                   hair_csv_path: str = './norm_region_hair_amount.csv', multiple: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads or extracts features.

    If there is a csv with "csv_path" loads that, if not extracts features from images at "images_path" and saves it as csv at "csv_path"

    Parameters
    ----------
    csv_path : str
        Path to features csv
    images_path : str
        Path to images
    metadata_path : str
        Path to dataset metadata
    features : list[Callable[[Image], float]]
        List of feature extraction functions, should be ordered as [feat_A, feat_B, ..., feat_n]
    hair_csv_path : str, optional
        Path to precomputed hair ratios, by default './norm_region_hair_amount.csv'
    multiple : bool, optional
        True if multiple classification, by default False

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (All features, All labels, Complete dataset)
    """
    try:
        data_df = pd.read_csv(csv_path).dropna()
    except FileNotFoundError:
        images: list[Image] = importImages(images_path, metadata_path, masks_path)
        data_df = extractFeatures(images, features, hair_csv_path)
        data_df.to_csv(csv_path, index=False)
        data_df = pd.read_csv(csv_path).dropna()

    # select only the baseline features
    baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    x_all = data_df[baseline_feats]
    y_all = data_df["diagnostic"] if multiple else data_df["true_melanoma"]

    return x_all, y_all, data_df