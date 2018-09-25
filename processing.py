import os
import numpy as np
import h5py
from tqdm import tqdm
import logging


def get_images_list(path_to_data, path_to_saved_file=None, dataset='img_names', **kwargs):
    '''Get the list of images to be processed'''
    if path_to_saved_file:
        data_file = h5py.File(path_to_saved_file, mode='r')
        processed_imgs = data_file[dataset]
    else:
        processed_imgs = []
        
    imgs = os.listdir(path_to_data)
    images = list(set(imgs)-set(processed_imgs))
    
    return images


def operate_on_chunks(pipeline, chunk_size, path_to_data, **kwargs):
    '''Apply operations from pipeline on chuncks'''
    images = get_images_list(path_to_data, **kwargs)
    nb_chunks = int(np.ceil(len(images)/chunk_size))
    
    for i in tqdm(range(nb_chunks)):
        chunk_images = images[i * chunk_size : i * chunk_size + chunk_size]
        chunk=chunk_images[:]
        for operation in pipeline:
            chunk = operation(chunk, path_to_data=path_to_data, chunk_images=chunk_images, **kwargs)
            
            
def process_dataset(categories, pipeline, path_to_data, chunk_size, **kwargs):
    '''Process the images in dataset according to pipeline'''
    for idx, category in enumerate(categories):
        logger = logging.getLogger(__name__)
        logger.info(f'Processing images from {category} category...')
        operate_on_chunks(pipeline=pipeline,
                          chunk_size=chunk_size,
                          path_to_data=os.path.join(path_to_data, category),
                          category=category,
                          **kwargs)