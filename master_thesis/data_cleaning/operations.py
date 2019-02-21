import os
import abc
import logging
from collections import namedtuple
import h5py
import numpy as np
from keras.applications import resnet50
from keras.preprocessing import image
from utils import get_labels

Chunk = namedtuple("Chunk", ["data", "image_name"])

class Operation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, chunk):
        pass


class LoadImages(Operation):
    """Load images to numpy array"""

    def __init__(self, path_to_images, img_size=(224, 224)):
        self.path_to_images = path_to_images
        self.img_size = img_size

    def __call__(self, chunk, **kwargs):
        _, img_names = chunk
        data, names = [], []

        for img_name in img_names:
            img_path = os.path.join(self.path_to_images, img_name)
            try:
                img = image.load_img(img_path, target_size = self.img_size)
                data.append(image.img_to_array(img))
                names.append(img_name)
            except:
                logger = logging.getLogger(__name__)
                logger.error(f"{img_name} loading error")

        return Chunk(np.array(data), names)


class ExtractFeatures(Operation):
    """Extract features for chunk"""

    def __init__(self, weights='imagenet', include_top=False, pooling='avg'):
        self.preprocess = resnet50.preprocess_input
        self.model = resnet50.ResNet50(weights = weights,
                                       include_top = include_top,
                                       pooling = pooling)

    def __call__(self, chunk, **kwargs):
        data, img_names = chunk
        data = self.preprocess(data)
        data = self.model.predict(data, batch_size = len(data))

        return Chunk(data, img_names)


class SaveFeatures(Operation):
    """Save the data to file_name"""

    def __init__(self, file_name, category_index, nb_features=2048, dtype='float32'):
        self.file_name = file_name
        self.nb_features = nb_features
        self.dtype = dtype
        self.category_index = category_index
        self.dt_str = h5py.special_dtype(vlen=str)

        self.prepare_file()

    def prepare_file(self):
        """Prepare hdf5 file for saving data"""
        with h5py.File(self.file_name, mode='a') as data_file:
            if 'features' not in data_file.keys():
                data_file.create_dataset('features', shape = (0, self.nb_features),
                                         dtype = self.dtype, maxshape = (None, self.nb_features))
            if 'image_names' not in data_file.keys():
                data_file.create_dataset('image_names', shape = (0,), dtype = self.dt_str, maxshape=(None,))
            if 'labels' not in data_file.keys():
                data_file.create_dataset('labels', shape = (0,), dtype = int, maxshape = (None,))

    def __call__(self, chunk, **kwargs):
        data, img_names = chunk
        labels = get_labels(img_names, self.category_index)

        with h5py.File(self.file_name, mode='a') as data_file:
            data_file['features'].resize((len(data_file['features']) + len(data)), axis=0)
            data_file['features'][-len(data):] = data

            data_file['image_names'].resize((len(data_file['image_names']) + len(img_names)), axis=0)
            data_file['image_names'][-len(img_names):] = img_names

            data_file['labels'].resize((len(data_file['labels']) + len(labels)), axis=0)
            data_file['labels'][-len(labels):] = labels

        return Chunk(data, img_names)


class ClassifyImages(Operation):
    """Predict the category of an image with defined model"""

    def __init__(self, model):
        self.model = model

    def __call__(self, chunk, **kwargs):
        data, img_names = chunk
        data = self.model.predict(data)

        return Chunk(data, img_names)


class SelectImages(Operation):
    """Remove images from redundant category"""

    def __init__(self, path_to_images, redundant_category):
        self.path_to_images = path_to_images
        self.redundant_category = redundant_category

    def __call__(self, chunk, **kwargs):
        data, img_names = chunk
        for idx, img_name in enumerate(img_names):
            if data[idx] == self.redundant_category:
                logger = logging.getLogger(__name__)
                logger.info(f"{img_name} removed")
                os.remove(os.path.join(self.path_to_images, img_name))

        return Chunk(data, img_names)
