import os
import numpy as np
import cv2
from keras.preprocessing import image

class ResizeLoader:
    """Loads images resizing them"""

    def __init__(self, dataset_path, img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size

    def __call__(self, image_list):
        images = []
        for image_name in image_list:
            img_path = os.path.join(self.dataset_path, image_name)
            img = image.load_img(img_path, target_size=self.img_size)
            img = np.asarray(img, dtype="uint8" )
            images.append(img)

        return np.array(images)


class PadLoader:
    """Loads images padding them"""

    def __init__(self, dataset_path, img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size

    def pad_image(self, img_path):
        img = cv2.imread(img_path)
        hight, width, _ = img.shape
        pad_width = int(max(0, hight-width)/2)
        pad_hight = int(max(0, width-hight)/2)
        img = cv2.copyMakeBorder(img, pad_hight, pad_hight, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __call__(self, image_list):
        images = []
        for image_name in image_list:
            img_path = os.path.join(self.dataset_path, image_name)
            img = self.pad_image(img_path)
            images.append(img)

        return np.array(images)
