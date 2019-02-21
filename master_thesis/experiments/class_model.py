import os
import numpy as np
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score

class ClassModel:
    """The instance of classification model"""

    def __init__(self, batch_size, model_dir=None):
        self.model_dir = model_dir
        self.model = self.__class__.get_model()

    @staticmethod
    def get_model():
        """Loads the model architecture"""
        input_tensor = Input(shape=(224, 224, 3))
        base_model = ResNet50(weights='imagenet', include_top=False)
        for layer in base_model.layers:
            layer.trainable=True
        x = base_model.output
        x = GlobalAveragePooling2D(data_format='channels_last')(x)
        x = Dense(9, activation='softmax')(x)
        updatedModel = Model(base_model.input, x)
        updatedModel.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

        return updatedModel

    def train(self, train_generator, valid_generator, batch_size, num_epochs):
        """Trains classification model"""
        callbacks = [ReduceLROnPlateau(min_lr = 0.00001)]
        if self.model_dir:
            callbacks.append(TensorBoard(log_dir = self.model_dir, batch_size = batch_size, write_graph = False))
            callbacks.append(ModelCheckpoint(os.path.join(self.model_dir, 'weights.h'), save_best_only = True))

        self.model.fit_generator(generator = train_generator,
                                 validation_data = valid_generator,
                                 epochs = num_epochs,
                                 callbacks = callbacks)

    def load_best_checkpoint(self):
        """Loads the best checkpoint from training"""
        model = self.__class__.get_model()
        model.load_weights(os.path.join(self.model_dir, 'weights.h'))
        self.model = model

    def predict(self, generator):
        """Predicts the class scores based on data from generator"""
        scores = self.model.predict_generator(generator = generator)
        return scores

    def evaluate(self, generator):
        """Evaluates the model"""
        metrics = {}
        scores = self.predict(generator)
        true_labels = generator.labels[:len(scores)]
        true_bin_labels = to_categorical(true_labels, num_classes=9)

        pred_bin = np.zeros((len(scores), 9))
        for idx, obs in enumerate(scores):
            pred_bin[idx][np.argmax(obs)] = 1

        metrics['accuracy'] = accuracy_score(true_bin_labels, pred_bin)
        metrics['auc'] = roc_auc_score(true_bin_labels, pred_bin)

        pred_label = np.zeros((len(scores)))
        for idx, obs in enumerate(scores):
            pred_label[idx] = np.argmax(obs)

        metrics['kappa'] = cohen_kappa_score(true_labels, pred_label)

        return metrics
