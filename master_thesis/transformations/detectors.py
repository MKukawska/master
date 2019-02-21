import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops

from utils import load_model, load_labels_map, rescale_box

class ObjectDetection:
    """The instance of model detecting objects"""

    def __init__(self, pb_model, labels_map, nb_of_classes):
        self.detection_graph = load_model(pb_model)
        self.category_index = load_labels_map(labels_map, nb_of_classes)

    def detect_boxes(self, batch):
        """Finds bounding boxes for images from batch"""
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        labels = self.detection_graph.get_tensor_by_name('detection_classes:0')

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                boxes, scores, labels = sess.run([boxes, scores, labels],
                                                 feed_dict={image_tensor: batch})

        return boxes, scores, labels

    def select_boxes(self, boxes, labels, scores, img_size, min_score):
        """Removes boxes under threshold"""
        bounding_boxes = {}
        predictions = zip(boxes, scores, labels)

        for box, score, label in predictions:
            if score > min_score:
                label = int(label)
                category = self.category_index[label]['name']
                ymin, xmin, ymax, xmax = rescale_box(box, img_size)

                bounding_boxes.setdefault(category, {'boxes': [], 'scores': []})
                bounding_boxes[category]['boxes'].append([xmin, xmax, ymin, ymax])
                bounding_boxes[category]['scores'].append(score)

        return bounding_boxes

    def predict(self, batch, min_score=0.3):
        """Returns the predicted bounding boxes"""
        img_size = batch.shape[1:3]
        boxes, scores, labels = self.detect_boxes(batch)

        batch_boxes = []
        for idx in range(len(batch)):
            bounding_boxes = self.select_boxes(boxes[idx], labels[idx], scores[idx], img_size, min_score)
            batch_boxes.append(bounding_boxes)

        return batch_boxes


class SegmentDetection:
    """The instance of model detecting segments"""

    def __init__(self, pb_model, labels_map, nb_of_classes):
        self.detection_graph = load_model(pb_model)
        self.category_index = load_labels_map(labels_map, nb_of_classes)

    def detect_segments(self, batch):
        """Finds segments for images from batch"""
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        num_detection = self.detection_graph.get_tensor_by_name('num_detections:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        labels = self.detection_graph.get_tensor_by_name('detection_classes:0')
        masks = self.detection_graph.get_tensor_by_name('detection_masks:0')

        all_detection_masks_reframed = []
        for idx, image in enumerate(batch):
            detection_boxes = tf.slice(boxes[idx], [0, 0], [100, -1])
            detection_masks = tf.slice(masks[idx], [0, 0, 0], [100, -1, -1])
            detection_masks_reframed = (utils_ops.reframe_box_masks_to_image_masks(detection_masks,
                                                                                   detection_boxes,
                                                                                   image.shape[0],
                                                                                   image.shape[1]))
            detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            all_detection_masks_reframed.append(tf.expand_dims(detection_masks_reframed, 0))
        masks = tf.concat(all_detection_masks_reframed, 0)

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                boxes, masks, scores, labels = sess.run([boxes, masks, scores, labels],
                                                 feed_dict={image_tensor: batch})

        return boxes, masks, scores, labels

    def select_masks(self, boxes, masks, labels, scores, img_size, min_score):
        """Removes masks under threshold"""
        segments = {}
        predictions = zip(boxes, masks, scores, labels)

        for box, mask, score, label in predictions:
            if score > min_score:
                label = int(label)
                category = self.category_index[label]['name']
                ymin, xmin, ymax, xmax = rescale_box(box, img_size)

                segments.setdefault(category, {'boxes': [], 'scores': [], 'masks': []})
                segments[category]['boxes'].append([xmin, xmax, ymin, ymax])
                segments[category]['scores'].append(score)
                segments[category]['masks'].append(mask)

        return segments

    def predict(self, batch, min_score=0.3):
        """Returns the predicted segments"""
        img_size = batch.shape[1:3]
        boxes, masks, scores, labels = self.detect_segments(batch)

        batch_segments = []
        for idx in range(len(batch)):
            prediction = (boxes[idx], masks[idx], labels[idx], scores[idx])
            segments = self.select_masks(boxes[idx], masks[idx], labels[idx], scores[idx], img_size, min_score)
            batch_segments.append(segments)

        return batch_segments
