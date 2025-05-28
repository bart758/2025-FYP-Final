import cv2 as cv
import numpy as np
from .image import Image

def find_midpoint_v1(image: np.ndarray) -> tuple[float, float]:
    """Finds midpoint in image

    Parameters
    ----------
    image : np.ndarray
        Image array

    Returns
    -------
    tuple[float, float]
        (row_mid, col_mid)
    """
    
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

def saveImageFile(img_rgb: np.ndarray, file_path: str):
    """Saves image to file.

    Parameters
    ----------
    img_rgb : np.ndarray
        Array containing image data.
    file_path : str
        Save path for image
    """
    try:
        # convert BGR
        img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

        # save the image
        success = cv.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save the image to {file_path}")
        return success

    except Exception as e:
        print(f"Error saving the image: {e}")
        return False

def get_hair_ratio(image: Image | np.ndarray, cfg=None) -> float:
    """
    Calculates the ratio of hair pixels in a single image.

    Parameters
    ----------
    image : Image
        Custom Image object with a `.color` attribute (numpy array).
    cfg : object, optional
        Configuration object with detection thresholds.

    Returns
    -------
    float
        Ratio of pixels detected as hair (hair_pixels / total_pixels).
    """

    if cfg is None:
        class CFG:
            def __init__(self):
                self.edge_low_threshold = 100
                self.edge_high_threshold = 220
                self.dark_spot_threshold = 150
                self.linelength_threshold = 10
                self.divergence_threshold = 0.25
                self.patchiness_threshold = 0.15

        cfg = CFG()

    if isinstance(image, Image):
        img_orig = image.color
    elif isinstance(image, np.ndarray):
        img_orig = image
    
    if img_orig.ndim == 3:
        img = img_orig.mean(-1).astype(np.uint8)
    else:
        img = img_orig.astype(np.uint8)

    image_size = img.shape[:2]
    total_pixels = image_size[0] * image_size[1]

    # Blackhat to enhance hair features
    kernel = np.ones((3, 3), np.uint8)
    img_filt = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    img_filt = np.where(img_filt > 15, img_filt, 0)

    kernel = np.ones((4, 4), np.uint8)
    img_filt = cv.morphologyEx(img_filt, cv.MORPH_DILATE, kernel)

    # Keep only darker regions (exclude brighter non-hair areas)
    dark_spots = (img < cfg.dark_spot_threshold).astype(np.uint8)
    dark_spots = cv.morphologyEx(dark_spots, cv.MORPH_DILATE, kernel)
    img_filt = img_filt * dark_spots

    # Detect line segments (potential hairs)
    lines = cv.HoughLinesP(img_filt, cv.HOUGH_PROBABILISTIC, np.pi / 90, 20, None, 1, 20)

    if lines is None:
        return 0.0

    lines = lines.reshape(-1, 4)
    Mask = np.zeros(image_size, dtype=np.uint8)

    for x1, y1, x2, y2 in lines:
        line_length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if line_length >= cfg.linelength_threshold:
            x_vals = np.linspace(x1, x2, int(line_length)).astype(int)
            y_vals = np.linspace(y1, y2, int(line_length)).astype(int)
            Mask[y_vals, x_vals] = 1

    # Slight dilation to connect nearby hair pixels
    kernel = np.ones((3, 3), np.uint8)
    Mask = cv.morphologyEx(Mask, cv.MORPH_DILATE, kernel)

    # Filter by patchiness and divergence to reduce false positives
    i, j = np.where(Mask != 0)
    if i.size == 0:
        return 0.0

    x_patchiness = np.std(j) / Mask.shape[1]
    y_patchiness = np.std(i) / Mask.shape[0]
    x_divergence = abs(0.5 - np.mean(i) / Mask.shape[0])
    y_divergence = abs(0.5 - np.mean(j) / Mask.shape[1])
    patchiness = np.sqrt(x_patchiness * y_patchiness)
    divergence = max(x_divergence, y_divergence)

    if divergence < cfg.divergence_threshold and patchiness < cfg.patchiness_threshold:
        return 0.0

    hair_pixels = np.count_nonzero(Mask)
    return hair_pixels / total_pixels
