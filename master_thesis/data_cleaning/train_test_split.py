import json
import os
from tqdm import tqdm

info_data_path = '/data/info_data/'
path_test = '/data/test/'
path_train = '/data/train/'
path_valid = '/dataset/valid/'
sample_size = 1000

def decode_image_data(data_dict, image, kept_images):
    """Decodes data for one image"""
    if image['files']:
        img_name = image['files'][0]['path'].split('/')[1]
        if img_name in kept_images:
            offer = data_dict.setdefault(image['offer_id'], [])
            offer.append(img_name)

def decode_brand_data(brand, images_path, info_data_path):
    """Decodes data for the brand"""
    data_dict = {}
    kept_images = os.listdir(os.path.join(images_path, brand))
    with open(os.path.join(info_data_path, f"{brand}.json"), 'r') as file:
        data = json.load(file)

    for image in tqdm(data):
        decode_image_data(data_dict, image, kept_images)

    return data_dict

def split_data(data_dict, offer_list, path_one, path_two, brand, sample_size, seed=123):
    """Splits data based on offers"""
    while len(os.listdir(os.path.join(path_two, brand))) < sample_size:
        np.random.seed(seed)
        idx = np.random.randint(0, high=len(offer_list))
        take_offer = offer_list[idx]

        for image in data_dict[take_offer]:
            os.rename(os.path.join(path_one, brand, image),
                      os.path.join(path_two, brand, image))
        offer_list.remove(take_offer)

    return offer_list


def main():
    for brand in os.listdir(path_train):
        data_dict = decode_brand_data(brand, path_train, info_data_path)
        offer_list = list(data_dict.keys())
        offer_list = split_data(data_dict, offer_list, path_train, path_test, brand, sample_size)
        offer_list = split_data(data_dict, offer_list, path_train, path_valid, brand, sample_size)

if __name__ == '__main__':
    sys.exit(main())