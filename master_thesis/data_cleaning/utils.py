import os
import pickle

def get_labels(img_names, category_index):
    """Extracts labels from images names"""
    labels = []
    for name in img_names:
        name = os.path.split(name)[0]
        label = category_index[name]
        labels.append(label)
        
    return labels

def transform_data(data, pipeline):
    """Transforms data"""
    for func in pipeline:
        data = func(data)
    return data

def save_model(file_name, model):
    """Save trained model to pickle file"""
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name):
    """Load trained model from pickle file"""
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model

def list_subdir_imgs(path_to_images):
    """List all images from all subdirectories of given path"""
    directories = os.listdir(path_to_images)
    full_list = []

    for directory in directories:
        directory_path = os.path.join(path_to_images, directory)
        images = [os.path.join(directory, name) for name in os.listdir(directory_path)]
        full_list += images

    return full_list
