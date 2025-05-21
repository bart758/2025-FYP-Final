import random
import cv2
import numpy as np

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
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # save the image
        success = cv2.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save the image to {file_path}")
        return success

    except Exception as e:
        print(f"Error saving the image: {e}")
        return False
