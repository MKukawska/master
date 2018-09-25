from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import pandas as pd

import pickle
import h5py
import numpy as np


def read_data(file_name, dataset):
    '''Read data from hdf5 file'''
    data_file = h5py.File(file_name, mode='r')
    return np.array(data_file[dataset])


def save_model(file_name, model):
    '''Save trained model to pickle file'''
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
        
        
def load_model(file_name):
    '''Load trained model from pickle file'''
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model


def reduce_features(features, pca_components, tsne_iter, random_state=314):
    '''Reduce features with PCA and TSNE to 2 components'''
    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=random_state, verbose=0, n_iter=tsne_iter)
    reduced_features = tsne.fit_transform(reduced_features)
    
    return reduced_features


def visualise_features(features, labels, pca_components, tsne_iter):
    '''Visualise features reduced to 2 components'''
    reduced_features = reduce_features(features, pca_components, tsne_iter)
    reduced_features = pd.DataFrame(reduced_features)
    reduced_features['class'] = labels
    plot = sns.pairplot(x_vars=[0], y_vars=[1], data=reduced_features, hue='class', size=10, markers="x")
    
    
def train_linear_svm(c_values, refit, n_jobs, x_train, y_train):
    '''Perform grid search for SVM and then retrain SVM'''
    parameters = {'C':c_values}
    linear_svc = svm.LinearSVC()
    svc = GridSearchCV(linear_svc, parameters, scoring='accuracy', refit=refit, n_jobs=n_jobs, verbose=0)
    svc.fit(x_train, y_train)
    return svc