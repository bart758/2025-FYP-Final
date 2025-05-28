from .img_util import get_hair_ratio
from .progressbar import progressbar
from .image import Image
import pandas as pd

def hair_import(images: list[Image], save_path: str, categorize: bool = False, 
                cat_thresholds: tuple[float, float] = (0.025933159722222224, 0.18871900770399305)) -> pd.DataFrame:
    """Compute hair ratio from list of Image objects.

    Uses the get_hair_ratio function to compute hair ratio and prepares the data for the hair_ratio function.

    Parameters
    ----------
    images : list[Image]
        List of Image objects.
    save_path : str
        Save path for final data csv
    categorize : bool, optional
        Change output to categorize the images innto 0, 1, 2 (no hair, some hair, a lot of hair), by default False
    cat_thresholds : tuple[float, float], optional
        Thresholds for the categorization 0 < t1, t2 < 2, by default (0.025933159722222224, 0.18871900770399305)

    Returns
    -------
    pd.DataFrame
        Computed ratios in column: Normalized or predicted_rating
    """
    def Categorize(data: pd.DataFrame, t1:float, t2:float) -> pd.DataFrame:
        data["predicted_rating"] = 2
        data.loc[data["Ratio"] <= t1, "predicted_rating"] = 0
        data.loc[(data["Ratio"] >= t1) & (data["Ratio"] <= t2), "predicted_rating"] = 1
        data.loc[data["Ratio"] >= t2, "predicted_rating"] = 2
        return data
    
    def normalize_row(row):
        mean = nb_summary.loc[row['Region'], ('Ratio', 'mean')]
        std = nb_summary.loc[row['Region'], ('Ratio', 'std')]
        return (row['Ratio'] - mean) / std

    numbers = pd.DataFrame()

    for i, image in progressbar(list(enumerate(images)), "Computing hair ratios: "):
        numbers.loc[i, 'ImageID'] = f"{image}"
        numbers.loc[i, 'Ratio'] = get_hair_ratio(image)
        numbers.loc[i, 'Region'] = image.metadata['region']
    numbers.set_index('ImageID', inplace=True)

    nb_summary = numbers.groupby(['Region']).describe()

    if categorize:
        numbers = Categorize(numbers, *cat_thresholds)
    else:
        numbers['Normalized'] = numbers.apply(normalize_row, axis=1)

    numbers = numbers.drop(columns=['Ratio', 'Region'])
    
    numbers.to_csv(save_path)

    return numbers
