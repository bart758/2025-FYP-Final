from matplotlib.pyplot import imshow
import cv2
import pandas as pd
import numpy as np
import os

def readImageFile(file_path):
    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray

class Image():

    # Class-level cache for metadata
    _metadata_path = None
    _metadata_df = None

    @classmethod
    def set_metadata_path(cls, metadata_path: str):
        cls._metadata_path = metadata_path     

    @classmethod
    def load_metadata(cls, csv_path: str):
        """Load metadata once and cache it."""
        if cls._metadata_df is None:
            if cls._metadata_path is None: 
                raise ValueError("Metadata path not loaded. Use Image.set_metadata_path first.")
            cls._metadata_df = pd.read_csv(csv_path, sep=',').set_index('img_id')


    def __init__(self, image_path: str):
        self.image_id = os.path.basename(image_path).split('/')[-1]
        self._image =  readImageFile(image_path)
        self.color = self._image[0]
        self.gray = self._image[1]
        self._metadata = None

    @property
    def metadata(self):
        """Load metadata only when accessed."""
        if Image._metadata_df is None:
            Image.load_metadata(Image._metadata_path)
        if self._metadata is None:
            self._metadata = Image._metadata_df.loc[self.image_id]
        return self._metadata
    
    @property
    def mask(self):
        try:
            return np.where(cv2.cvtColor(cv2.imread("".join(["masks\\", self.image_id.split(".")[0], "_mask.png"])), cv2.COLOR_BGR2GRAY) > 10, 1, 0)
        except:
            raise FileNotFoundError(f"Mask for {self.image_id} not found in .masks/ directory.")
    
    def __lt__(self, other):
        return int(self.metadata["patient_id"][4:]) < int(other.metadata["patient_id"][4:])
    
    def __eq__(self, other):
        return int(self.metadata["patient_id"][4:]) == int(other.metadata["patient_id"][4:])
    
    def __str__(self):
        return self.metadata.name

    def __repr__(self):
        return self.metadata.name




    