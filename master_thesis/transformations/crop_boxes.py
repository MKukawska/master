import sys
import argparse
import logging
from tqdm import tqdm

from batch_iterator import BatchIterator
from detectors import ObjectDetection
from operations import GetCategoryBox, CropBoxes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pb_model = './models/boxes/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb'
labels_map = './models/mscoco_label_map.pbtxt'

category = 'car'
nb_of_classes = 90
batch_size = 5

def main(args):
    img_path = f"/data/{args.dataset}/original"
    cropped_path = f"/data/{args.dataset}/cropped"

    iterator = BatchIterator(img_path, batch_size, continue_path = cropped_path)
    detector = ObjectDetection(pb_model, labels_map, nb_of_classes)
    get_category = GetCategoryBox(category, img_path)
    crop_boxes = CropBoxes(img_path, cropped_path)

    for batch, names in tqdm(iterator):
        boxes = detector.predict(batch)
        main_boxes, images = get_category(boxes, names)
        crop_boxes(main_boxes, images)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='train, test or valid')
    args = parser.parse_args()

    sys.exit(main(args))
