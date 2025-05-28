import pandas as pd
from collections.abc import Callable
from util.progressbar import progressbar
from util.hair_feature_util import hair_import
from .Feature_F import hair_ratio
from .image import Image

def extractFeatures(images: list[Image], extraction_functions: list[Callable[..., float]], hair_csv_path: str = './norm_region_hair_amount.csv') -> pd.DataFrame:
    """Extracts features from list of images using funtions from extraction_functions list and saves them into a dataframe.

    Args:
        images (list[Image]): list of Image objects
        extraction_functions (list[Callable[..., float]]): List of feature extraction functions, should be ordered as [feat_A, feat_B, ..., feat_n]

    Returns:
        pd.DataFrame: Columns "patient_id" | "feat_A" | "feat_B" | ... | "feat_n" | "true_melanoma"
    """
    if hair_ratio in extraction_functions:
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
                
        features_df.loc[i_image, "true_melanoma"] = True if image.metadata["diagnostic"] == "MEL" else False

    print(f"There was an error proccessing {counter} images.")
    return features_df
