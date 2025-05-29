import numpy as np
from sklearn.cluster import  MiniBatchKMeans
from itertools import combinations
from .image import Image

def get_multicolor_rate(image: Image, n: int = 4) -> float:
    """
    Measure color variation in the masked lesion area using clustering.
    Returns normalized color variation in [0, 1] range.

    Parameters
    ----------
    image: np.ndarray
        Color image of the lesion.
    mask: np.ndarray
        Mask of the lesion image.
    n: int, optional
        Number of color groups. (Conservative: 3, Sensitive: 5), by default 4.

    Returns
    -------
    float
        Normalized maximum distance between colors in lesion.
    """
    mask = image.mask_cropped
    image = image.image_cropped

    col_list = image[mask != 0].reshape(-1, 3)

    if len(col_list) < 2:
        return 0.0

    # KMeans clustering (used MinibatchKMeans since it is significantly faster)
    cluster = MiniBatchKMeans(n_clusters=n, n_init=10, random_state=0).fit(col_list)
    centers = cluster.cluster_centers_

    # Calculate max pairwise distance
    max_dist = max(np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in combinations(centers, 2))

    # Normalize RGB values to [0, 1]
    normalized = round(max_dist / np.sqrt(3), 4)

    return normalized