from .image import Image

"""There is no way PAT_456_888_961 is the largest one"""
def find_max_diameter(image: Image)->float:
    """Gets diameter of a lesion from the metadata.

    Parameters
    ----------
    image : Image
        Image object

    Returns
    -------
    float
        Max of horizontal and vertical diameter of the lesion from metadata.
    """
    return max(image.metadata['diameter_1'],image.metadata['diameter_2'])
