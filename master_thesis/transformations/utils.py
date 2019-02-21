import os
import tensorflow as tf
from object_detection.utils import label_map_util

def load_model(pb_model):
    """Loads frozen detection model"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return graph

def load_labels_map(labels_map, nb_of_classes):
    """Loads labels map and return category_index"""
    label_map = label_map_util.load_labelmap(labels_map)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=nb_of_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index

def rescale_box(box, img_size):
    """Rescale raw boxes"""
    ymin = box[0] * img_size[0]
    xmin = box[1] * img_size[1]
    ymax = box[2] * img_size[0]
    xmax = box[3] * img_size[1]

    return (ymin, xmin, ymax, xmax)

def list_subdir_imgs(path_to_images):
    """List all images from all subdirectories of given path"""
    directories = os.listdir(path_to_images)
    full_list = []

    for directory in directories:
        directory_path = os.path.join(path_to_images, directory)
        images = [os.path.join(directory, name) for name in os.listdir(directory_path)]
        full_list += images

    return full_list
