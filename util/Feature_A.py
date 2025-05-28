import numpy as np
from .image import Image
from .img_util import find_midpoint_v1


def asymmetry(image: Image)->float:
    """Computes asymetry in the binary mask by comparing upper/lower and rigtg/left halves.

    Measures the asymmetry of an image. The function splits the image into upper/lower halves and right/left halves.
    Then flips lower and right halves. Then it uses xor to compare the halfs.
    Creating arrays which have values of 1 where compared halves are 
    not the same. Then it calculates the actual score by summing all the not symmetrical 
    pixels and dividing their sum by 2 times the sum of all the pixels of the original mask.

    Parameters
    ----------
    image : Image
        Image object

    Returns
    -------
    float
        Ratio of asymetry
    """
    mask = image.mask_cropped
    row_mid, col_mid = find_midpoint_v1(mask)

    upper_half = mask[:int(np.ceil(row_mid)), :]
    lower_half = mask[int(row_mid):, :]
    left_half = mask[:, :int(np.ceil(col_mid))]
    right_half = mask[:, int(col_mid):]

    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return asymmetry_score