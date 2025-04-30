import img_util as util
import pandas as pd
import os

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
        self._image =  util.readImageFile(image_path)
        self.color = self._image[0]
        self.gray = self._image[1]
        self._metadata = None

    @property
    def metadata(self):
        """Load metadata only when accessed."""
        if Image._metadata_df is None:
            Image.load_metadata(Image._metadata_path)
        if self._metadata is None:
            self._metadata = Image._metadata_df.loc[self.image_id].to_dict()
        return self._metadata




    