import numpy as np
from .image import Image

def is_growing(image: Image) -> bool | None:
    """Gets "grow" from metadata.

    Parameters
    ----------
    image : Image
        Image object

    Returns
    -------
    bool | None
        True if "grow" true in metadata, False if false. None when Unknown.
    """
    if image.metadata["grew"] == "TRUE":
        return True
    elif image.metadata["grew"] == "FALSE":
        return False
    else:
        return None
