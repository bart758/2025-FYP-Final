import cv2
import pandas as pd
import numpy as np
import os
from .progressbar import progressbar

def readImageFile(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    # read image as an 8-bit array
    img_bgr: np.ndarray = cv2.imread(file_path)

    # convert to RGB
    img_rgb: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray: np.ndarray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray

def cut_mask(mask: np.ndarray, image_id: str = "") -> np.ndarray:
    """Cuts empty / excess borders.

    Isolates the area of interest. Removes all borders 
    (rows/columns) that sum up to 0. Basically making a rectangle around the area of interest
    and cutting it.

    Parameters
    ----------
    mask : np.ndarray
        Mask of lession
    image_id: str, optional
        Id of image mask

    Returns
    -------
    np.ndarray
        _description_
    """
    
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    if not active_cols or not active_rows:
        raise ValueError(f"Mask for image {image_id} is empty.")

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_

def cut_im_by_mask(image: np.ndarray, mask: np.ndarray, image_id: str = "") -> np.ndarray:
    """Cuts empty / excess borders of image by mask.

    The function masks the active columns / rows based on the mask and then crops the 
    image based on that. So the returned image will be a rectangle just zoomed in on lession
    based on the mask.

    Parameters
    ----------
    image : _type_
        _description_
    mask : _type_
        _description_
    image_id: _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    if not active_cols or not active_rows:
        raise ValueError(f"Mask for image {image_id} is empty.")

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_image = image[row_min:row_max+1, col_min:col_max+1]

    return cut_image

class Image():
    """Image object class

    Supports lt, eq based on patien_id.

    Atributes:
    ---------
        _metadata_path: str:
            Path to metadata csv.
        _metadata_df: pd.DataFrame:
            Pandas dataframe of the metadata csv.

    Raises:
        ValueError: If metadata_path not set when metadata is attempted to be loaded.
        FileNotFoundError: Mask for image not found.
    """

    # Class-level cache for metadata
    _metadata_path: str = None
    _metadata_df: pd.DataFrame = None

    @classmethod
    def set_metadata_path(cls, metadata_path: str):
        """Initialize class atribute _metadata_path for use in Image.load_metadata()

        Args:
            metadata_path (str): Path to metadata csv
        """
        cls._metadata_path = metadata_path     

    @classmethod
    def load_metadata(cls, csv_path: str):
        """Initialize class atribute _metadata_df with metadata from metadata csv

        Args:
            csv_path (str): Path to metadata csv

        Raises:
            ValueError: if metadata path is not set, use Image.set_metadata_path("metadata_path") to fix.
        """
        if cls._metadata_df is None:
            if cls._metadata_path is None: 
                raise ValueError("Metadata path not loaded. Use Image.set_metadata_path first.")
            cls._metadata_df = pd.read_csv(csv_path, sep=',').set_index('img_id')


    def __init__(self, image_path: str):
        """Initialize Image object from image file

        Parameters
        ----------
        image_path : str
            Path to image file to be loaded.
        """
        self.image_id: str = os.path.basename(image_path).split('/')[-1]
        self._image =  readImageFile(image_path)
        self.color: np.ndarray = self._image[0]
        self.gray: np.ndarray = self._image[1]
        self._metadata = None

    @property
    def metadata(self) -> pd.Series:
        """Load metadata when accessed

        Returns
        -------
        pd.Series
            Row from metadata_df corresponding to image
        """
        if Image._metadata_df is None:
            Image.load_metadata(Image._metadata_path)
        if self._metadata is None:
            self._metadata = Image._metadata_df.loc[self.image_id]
        return self._metadata
    
    @property
    def mask(self) -> np.ndarray:
        """Load mask corresponding to Image file in .masks/ directory

        Returns
        -------
        np.ndarray
            Binary mask array.

        Raises
        ------
        FileNotFoundError
            If image does not have a mask in the .masks/ directory.
        """
        try:
            return np.where(cv2.imread("".join(["masks/", self.image_id.split(".")[0], "_mask.png"]), cv2.IMREAD_GRAYSCALE) >= 1, 1, 0).astype(np.uint8)
        except:
            raise FileNotFoundError(f"Mask for {self.image_id} not found in .masks/ directory.")
        
    @property
    def mask_cropped(self) -> np.ndarray:
        """Mask cropped to only include rows and columns that have a one.

        Returns
        -------
        np.ndarray
            Binary mask array.
        """
        return cut_mask(self.mask, str(self))
    
    @property
    def image_cropped(self) -> np.ndarray:
        """Image cut to only include the area where binary mask overlaps.

        Returns
        -------
        np.ndarray
            Image array.
        """
        return cut_im_by_mask(self.color, self.mask, str(self))
    
    @property
    def gray_cropped(self) -> np.ndarray:
        """Gray image cut to only include the area where binary mask overlaps.

        _extended_summary_

        Returns
        -------
        np.ndarray
            Gray image array.
        """
        return cut_im_by_mask(self.gray, self.mask)
    
    @property
    def hair_removed(self) -> np.ndarray:
        """Image with hair removed.

        NOT YET IMPLEMENTED

        Returns
        -------
        np.ndarray
            Image array with hair removed.
        """
        ...
    
    def __lt__(self, other):
        return int(self.metadata["patient_id"][4:]) < int(other.metadata["patient_id"][4:])
    
    def __eq__(self, other):
        return int(self.metadata["patient_id"][4:]) == int(other.metadata["patient_id"][4:])
    
    def __str__(self):
        return self.metadata.name

    def __repr__(self):
        return self.metadata.name

def importImages(directory: str, metadata_path: str) -> list[Image]:
    """Import all image files from directory.

    Parameters
    ----------
    directory : str
        Image files parent directory path.
    metadata_path : str
        Path to metadata csv

    Returns
    -------
    list[Image]
        List of Image objects containing all images from directory.
    """

    Image.set_metadata_path(metadata_path)

    file_list: list[str] = [os.path.join(directory, f) for f in os.listdir(directory) if
                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    images: list[Image] = []

    for image_path in progressbar(file_list, "Loading images: ", 40):
        images.append(Image(image_path))
    print("All images loaded succesfuly") 
    
    return images




    