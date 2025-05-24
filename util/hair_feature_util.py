from .img_util import hair_removed
from .image import Image
import pandas as pd
def hair_ratio(image: Image, hair_df: pd.DataFrame) -> float:
    return hair_df.loc[f'{image}', 'Normalized']


def hair_import(images: list[Image], save_path: str) -> pd.DataFrame:
    numbers = pd.DataFrame()

    for i, image in enumerate(images):
        numbers.loc[i, 'ImageID'] = image
        numbers.loc[i, 'Ratio'] = hair_removed(image)
        numbers.loc[i, 'Region'] = image.metadata['region']

    nb_summary = numbers.groupby(['Region']).describe()

    def normalize_row(row):
        mean = nb_summary.loc[row['Region'], ('Ratio', 'mean')]
        std = nb_summary.loc[row['Region'], ('Ratio', 'std')]
        return (row['Ratio'] - mean) / std

    numbers['Normalized'] = numbers.apply(normalize_row, axis=1)

    numbers = numbers.drop(columns=['Ratio', 'Region'])
    numbers.set_index('ImageID', inplace=True)

    numbers.to_csv(save_path)

    return numbers
