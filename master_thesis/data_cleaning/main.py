import sys
import logging
import os
from tqdm import tqdm

from image_manager import ImageManager
from operations import LoadImages, ExtractFeatures, SaveFeatures, ClassifyImages, SelectImages
from cleaning_model import CleaningModel
from utils import transform_data, get_labels, save_model, load_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


train_images_path = './data'
dataset_path = './data'
category_index = {'auto': 0, 'other': 1}
redundant_category = 1
features_file = './features/resnet.hdf5'
model_dir = './model'
chunk_size = 100
epochs = 10


def extract_resnet_features(images_path, category_index, features_file, chunk_size):
    """Run pipeline extracting resnet features"""
    image_manager = ImageManager(images_path, chunk_size, continue_file = features_file)
    pipeline = [LoadImages(images_path),
                ExtractFeatures(),
                SaveFeatures(features_file, category_index)]

    for chunk in tqdm(image_manager):
        chunk = transform_data(chunk, pipeline)

def train_model(images_path, features_file, category_index, model_dir, batch_size, epochs):
    """Train model for cleaning data"""
    image_manager = ImageManager(images_path, chunk_size = 1)
    image_manager.load_features(features_file)
    x = image_manager.data
    y = get_labels(image_manager.images, category_index)

    model = CleaningModel(model_dir)
    model.train(x, y, batch_size = batch_size, epochs = epochs)
    model.load_best_checkpoint()

    save_model(os.path.join(model_dir, 'model.p'), model)

    return model

def clean_data(model, images_path, chunk_size, redundant_category):
    """Run pipeline cleaning the data"""
    image_manager = ImageManager(images_path, chunk_size)
    pipeline = [LoadImages(images_path),
                ExtractFeatures(),
                ClassifyImages(model),
                SelectImages(images_path, redundant_category)]

    for chunk in tqdm(image_manager):
        chunk = transform_data(chunk, pipeline)


def main():
    extract_resnet_features(train_images_path, category_index, features_file, chunk_size)
    model = train_model(train_images_path, features_file, category_index, model_dir, chunk_size, epochs)
    clean_data(model, dataset_path, chunk_size, redundant_category)

if __name__ == '__main__':
    sys.exit(main())
