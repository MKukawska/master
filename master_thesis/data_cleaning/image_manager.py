import logging
import h5py
import numpy as np
from utils import list_subdir_imgs

class ImageManager:
    """Helps to manage the image dataset"""

    def __init__(self, images_path, chunk_size, continue_file=None):
        self.images_path = images_path
        self.continue_file = continue_file
        self.chunk_size = chunk_size
        self.images = self.get_valid_images()
        self.data = np.zeros(len(self.images))
        self.n_chunks = int(np.ceil(len(self.images) / self.chunk_size))

    def get_valid_images(self):
        """Gets the names of images that needs to be processed"""
        images = list_subdir_imgs(self.images_path)
        if self.continue_file:
            try:
                with h5py.File(self.continue_file, 'r') as data_file:
                    processed_images = set(data_file['image_names'])
                images = list(set(images).difference(processed_images))
            except OSError:
                logger = logging.getLogger(__name__)
                logger.warning(f"{self.continue_file} not found - processing all images under {self.images_path}")

        return images

    def load_features(self, features_file):
        """Loads already extracted features from hdf5"""
        data = []
        with h5py.File(features_file, 'r') as data_file:
            file_images = list(data_file['image_names'])
            for image in self.images:
                idx = file_images.index(image)
                data.append(data_file['features'][idx])
        self.data = np.array(data)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, i):
        if i >= self.n_chunks:
            raise IndexError(f"Only {self.n_chunks} chunks are available")
        low = i * self.chunk_size
        high = (i + 1) * self.chunk_size

        return (self.data[low:high], self.images[low:high])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
