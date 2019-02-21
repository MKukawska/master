import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.utils import to_categorical


class CleaningModel:

    def __init__(self, model_dir=None):
        self.model_dir = model_dir
        self.model = self.get_model()

    def get_model(self):
        """Loads cleaning model"""
        model_input = Input(shape = (2048, 1))
        x = model_input
        x = Flatten()(x)
        x = Dense(100, activation = "relu")(x)
        output = Dense(2, activation = "softmax")(x)
        model = Model(model_input, output)
        model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ["accuracy"])

        return model

    def train(self, x, y, batch_size, epochs, test_size=0.25, random_state=123):
        """Trains cleaning model"""
        y = to_categorical(y, num_classes = 2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)

        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

        callbacks = [ReduceLROnPlateau(min_lr = 0.00001)]
        if self.model_dir:
            callbacks.append(TensorBoard(log_dir = self.model_dir, batch_size = batch_size, write_graph = False))
            callbacks.append(ModelCheckpoint(os.path.join(self.model_dir, 'weights.h'), save_best_only = True))

        self.model.fit(x_train,
                       y_train,
                       batch_size = batch_size,
                       epochs = epochs,
                       callbacks = callbacks,
                       validation_data = (x_test, y_test))

    def load_best_checkpoint(self):
        """Loads the best checkpoint from training"""
        model = self.get_model()
        model.load_weights(os.path.join(self.model_dir, 'weights.h'))
        self.model = model

    def predict(self, data):
        """Predicts the class based on data"""
        data = data.reshape(data.shape + (1,))
        data = self.model.predict(data, batch_size = len(data))
        return [np.argmax(y) for y in data]
