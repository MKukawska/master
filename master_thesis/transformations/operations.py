import os
import logging
import numpy as np
from PIL import Image
import cv2

class GetCategoryBox:
    """Gets the biggest box from given category"""

    def __init__(self, category, img_path):
        self.category = category
        self.img_path = img_path

    @staticmethod
    def get_main_box(boxes):
        """Returns the biggest box"""
        if len(boxes) > 1:
            x1 = boxes[:, 0]
            x2 = boxes[:, 1]
            y1 = boxes[:, 2]
            y2 = boxes[:, 3]
            area = (x2 - x1) * (y2 - y1)
            main_box = boxes[np.argmax(area)]
        else:
            main_box = np.squeeze(boxes)

        return main_box

    def __call__(self, batch_boxes, batch_names):
        main_boxes = []
        images = []

        for boxes, name in zip(batch_boxes, batch_names):
            try:
                category_boxes = np.array(boxes[self.category]['boxes'])
                main_box = self.__class__.get_main_box(category_boxes)
                main_boxes.append(main_box)
                images.append(name)
            except KeyError:
                logger = logging.getLogger(__name__)
                logger.error(f"{name}: {self.category} not found")
                os.remove(os.path.join(self.img_path, name))

        return main_boxes, images


class CropBoxes:
    """Saves the bounding box for image"""

    def __init__(self, img_path, cropped_path):
        self.img_path = img_path
        self.cropped_path = cropped_path

    def crop_box(self, box, image):
        """Crops box from image"""
        img = Image.open(os.path.join(self.img_path, image))
        xmin, xmax, ymin, ymax = box
        cropped_box = img.crop((xmin, ymin, xmax, ymax))

        return cropped_box

    def __call__(self, boxes, images):
        for box, image in zip(boxes, images):
            cropped_box = self.crop_box(box, image)
            cropped_box.save(os.path.join(self.cropped_path, image))


class GetCategoryMask:
    """Gets the biggest mask from given category"""

    def __init__(self, category, img_path):
        self.category = category
        self.img_path = img_path

    @staticmethod
    def get_main_segment(segments):
        """Returns the biggest mask"""
        masks = segments['masks']
        boxes = np.array(segments['boxes'])

        if len(boxes) > 1:
            x1 = boxes[:, 0]
            x2 = boxes[:, 1]
            y1 = boxes[:, 2]
            y2 = boxes[:, 3]
            area = (x2 - x1) * (y2 - y1)
            main_mask = masks[np.argmax(area)]
            main_box = boxes[np.argmax(area)]
        else:
            main_mask = np.squeeze(masks)
            main_box = np.squeeze(boxes)

        return main_mask, main_box

    def __call__(self, batch_segments, batch_images, batch_names):
        main_masks = []
        main_boxes = []
        image_names = []

        for segments, image, name in zip(batch_segments, batch_images, batch_names):
            try:
                main_mask, main_box = self.__class__.get_main_segment(segments[self.category])
                main_mask = np.dstack([main_mask]*3)
                main_mask = np.multiply(image, main_mask)
                main_masks.append(main_mask)
                main_boxes.append(main_box)
                image_names.append(name)
            except KeyError:
                logger = logging.getLogger(__name__)
                logger.error(f"{name}: {self.category} not found")
                os.remove(os.path.join(self.img_path, name))

        return main_masks, main_boxes, image_names


class CropMasks:
    """Saves the mask for image"""

    def __init__(self, cropped_path):
        self.cropped_path = cropped_path

    @staticmethod
    def crop_mask(mask, box):
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(mask)
        xmin, xmax, ymin, ymax = box
        cropped_image = img.crop((xmin, ymin, xmax, ymax))

        return cropped_image

    def __call__(self, main_masks, main_boxes, image_names):
        for mask, box, image in zip(main_masks, main_boxes, image_names):
            cropped_mask = self.__class__.crop_mask(mask, box)
            cropped_mask.save(os.path.join(self.cropped_path, image))
