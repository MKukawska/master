import os
from PIL import Image
import numpy as np
from utils import list_subdir_imgs

class BatchIterator:
    """Iterates through images"""

    def __init__(self, img_path, batch_size, continue_path=None):
        self.img_path = img_path
        self.batch_size = batch_size
        self.images = self.get_valid_images(continue_path)
        self.n_batch = int(np.ceil(len(self.images) / self.batch_size))

    def get_valid_images(self, continue_path):
        """Gets the names of images that needs to be processed"""
        images = list_subdir_imgs(self.img_path)
        if continue_path:
            processed = list_subdir_imgs(continue_path)
            images = list(set(images).difference(processed))

        return images

    def load_images(self, batch_names):
        """Loads images to numpy arrays"""
        batch_imgs = []
        widths = []
        heights = []

        for image in batch_names:
            img = Image.open(os.path.join(self.img_path, image))
            batch_imgs.append(np.array(img))
            widths.append(img.size[0])
            heights.append(img.size[1])

        return batch_imgs, widths, heights

    def load_detection_batch(self, batch_names):
        """Loads images to the detection batch"""
        batch_imgs, widths, heights = self.load_images(batch_names)
        max_width = max(widths)
        max_height = max(heights)

        batch = np.full([len(batch_names), max_height, max_width, 3], 255, dtype=int)
        for idx, image in enumerate(batch_imgs):
            batch[idx][0:heights[idx], 0:widths[idx], 0:3] = image

        return batch

    def __len__(self):
        return self.n_batch

    def __getitem__(self, i):
        if i >= self.n_batch:
            raise IndexError(f"Only {self.n_batch} batches are available")
        low = i * self.batch_size
        high = (i + 1) * self.batch_size

        return (self.load_detection_batch(self.images[low:high]), self.images[low:high])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
