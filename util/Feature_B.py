import numpy as np
from skimage import measure
from .image import Image

def compactness_score(image: Image)->float:
    """Computes compactness score of binary mask by formula (4*pi*area)/(perimeter^2).

    Higher means more compact, closer to a circle.

    Parameters
    ----------
    image : Image
        Image object

    Returns
    -------
    float
        Compactness of the binary mask
    """
    mask = image.mask_cropped
    A = np.sum(mask) # calculates the area of the mask - 1 where white, 0 where black, sum is number of white pixels in mask - lession

    l = measure.perimeter_crofton(mask)
    
    compactness = (4*np.pi*A)/(l**2) # calculate compactness by formula
    
    return compactness
