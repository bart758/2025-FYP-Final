from .image import Image
import pandas as pd

def hair_ratio(image: Image, hair_df: pd.DataFrame) -> float:
    """Gets hair ratio for a given image from precomputed data.

    Parameters
    ----------
    image : Image
        Image object.
    hair_df : pd.DataFrame
        Precomputed ratio data, must include columns: Normalized or predicted_rating

    Returns
    -------
    float
        Hair ratio for given image
    """
    try:
        return hair_df.loc[f'{image}', 'Normalized']
    except KeyError as e:
        return hair_df.loc[f'{image}', 'predicted_rating']