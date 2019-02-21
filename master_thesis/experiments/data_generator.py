import os
import pickle
import numpy as np
import keras

class ImageGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, dataset_path, category_index_path, loader, batch_size, n_classes=9, shuffle=True):
        self.image_list = self.__class__.get_image_list(dataset_path)
        self.labels = self.get_labels(category_index_path)
        self.loader = loader
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    @staticmethod
    def get_image_list(dataset_path):
        '''Returns list of all images in dataset'''
        image_list = []
        for brand in os.listdir(dataset_path):
            images = os.listdir(os.path.join(dataset_path, brand))
            image_list += [os.path.join(brand, image_name) for image_name in images]

        return np.array(image_list)

    def get_labels(self, category_index_path):
        '''Returns labels'''
        with open(category_index_path, 'rb') as file:
            category_index = pickle.load(file)

        labels = []
        for image_name in self.image_list:
            label = os.path.split(image_name)[0]
            labels.append(category_index[label])

        return np.array(labels)

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        '''Generates data containing batch_size samples''' # X : (n_samples, *dim, n_channels)
        X = self.loader(self.image_list[indexes])
        y = self.labels[indexes]
        
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
