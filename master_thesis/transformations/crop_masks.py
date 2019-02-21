import sys
import argparse
import logging
from tqdm import tqdm

from batch_iterator import BatchIterator
from detectors import SegmentDetection
from operations import GetCategoryMask, CropMasks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pb_model = './models/masks/mask_rcnn_inception_v2_coco/frozen_inference_graph.pb'
labels_map = './models/mscoco_label_map.pbtxt'
category = 'car'
nb_of_classes = 90
batch_size = 5

def main(args):
    img_path = f"/data/{args.dataset}/original"
    segmented_path = f"/data/{args.dataset}/segmented"

    iterator = BatchIterator(img_path, batch_size, continue_path = segmented_path)
    detector = SegmentDetection(pb_model, labels_map, nb_of_classes)
    get_category = GetCategoryMask(category, img_path)
    crop_masks = CropMasks(segmented_path)

    for batch, names in tqdm(iterator):
        boxes = detector.predict(batch)
        main_masks, main_boxes, image_names = get_category(boxes, batch, names)
        crop_masks(main_masks, main_boxes, image_names)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='train, test or valid')
    args = parser.parse_args()

    sys.exit(main(args))
