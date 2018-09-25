import os
import abc
import h5py
from collections import namedtuple
import numpy as np
from keras.applications import resnet50
from keras.preprocessing import image

from scripts.utils import load_model


ModelPair = namedtuple("ModelPair", ["preprocess_input", "model"])
MODELS = {'resnet50': ModelPair(resnet50.preprocess_input, resnet50.ResNet50)}


class Operation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, chunk):
        pass


class Save(Operation):
    '''Save the data to file_name'''
    def __init__(self, file_name, dataset_name, shape, dtype, maxshape):
        self.data_file = h5py.File(file_name, mode='a')
        self.dataset_name = dataset_name
        if dataset_name not in self.data_file.keys():
            self.data_file.create_dataset(dataset_name, shape=shape, dtype=dtype, maxshape=maxshape)
        
    def __call__(self, chunk, **kwargs):
        self.data_file[self.dataset_name].resize((len(self.data_file[self.dataset_name]) + len(chunk)), axis = 0)
        self.data_file[self.dataset_name][-len(chunk):] = chunk
    
        return chunk
    
    
class LoadImages(Operation):
    '''Load images to numpy array'''
    def __init__(self, img_size=224, grayscale=False):
        self.img_size = img_size
        self.grayscale = grayscale
        self.nb_channels = 1 if grayscale else 3
    
    def __call__(self, chunk, path_to_data, **kwargs):
        loaded_chunk = np.empty([len(chunk), self.img_size, self.img_size, self.nb_channels])
        for idx, img_name in enumerate(chunk):
            img_path = os.path.join(path_to_data, img_name)
            img = image.load_img(img_path, target_size=(self.img_size, self.img_size), grayscale=self.grayscale)
            img = image.img_to_array(img)
            loaded_chunk[idx] = img
        
        return loaded_chunk


class NormImages(Operation):
    '''Normalize images from array'''
    def __call__(self, chunk, **kwargs):
        return chunk/255
    
    
class ExtractFeatures(Operation):
    '''Extract features for chunk'''
    def __init__(self, method='resnet50', **kwargs):
        preprocess, Model = MODELS[method]
        self.preprocess = preprocess
        self.model = Model(**kwargs)
        
    def __call__(self, chunk, **kwargs):
        chunk = self.preprocess(chunk)
        chunk = self.model.predict(chunk, batch_size=len(chunk))
        
        return chunk
    
    
class AppendLabels(Operation):
    '''Append labels for chunk'''
    def __init__(self, category_index):
        self.category_index = category_index
        
    def __call__(self, chunk, category, **kwargs):
        labels = [self.category_index[category]] * len(chunk)
        
        return labels
    
    
class ClassifyImages(Operation):
    '''Predict the category of an image with defined model'''
    def __init__(self, path_to_model):
        self.model = load_model(path_to_model)
    
    def __call__(self, chunk, **kwargs):
        chunk = self.model.predict(chunk)
        
        return chunk


class SelectImages(Operation):
    '''Remove images from redundant category'''
    def __init__(self, redundant_category):
        self.redundant_category = redundant_category
        
    def __call__(self, chunk, chunk_images, path_to_data, **kwargs):
        for idx, img_name in enumerate(chunk_images):
            if chunk[idx] == self.redundant_category:
                os.remove(os.path.join(path_to_data, img_name))

        return chunk