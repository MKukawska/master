import argparse
import os
import json

from loaders import ResizeLoader, PadLoader
from data_generator import ImageGenerator
from class_model import ClassModel

category_index = '/data/category_index.pickle'

def main(args):
    train_path = f'/data/train/{args.dataset}'
    valid_path = f'/data/valid/{args.dataset}'
    test_path = f'/data/test/{args.dataset}'
    model_dir = f'/models/{args.dataset}/{args.loader}'

    if args.loader == 'resize':
        train_generator = ImageGenerator(train_path, category_index, ResizeLoader(train_path), args.batch_size)
        valid_generator = ImageGenerator(valid_path, category_index, ResizeLoader(valid_path), args.batch_size, shuffle = False)
        test_generator = ImageGenerator(test_path, category_index, ResizeLoader(test_path), args.batch_size, shuffle = False)
    elif args.loader == 'pad':
        train_generator = ImageGenerator(train_path, category_index, PadLoader(train_path), args.batch_size)
        valid_generator = ImageGenerator(valid_path, category_index, PadLoader(valid_path), args.batch_size, shuffle = False)
        test_generator = ImageGenerator(test_path, category_index, PadLoader(test_path), args.batch_size, shuffle = False)

    model = ClassModel(batch_size, model_dir)
    model.train(train_generator, valid_generator, args.batch_size, args.num_epochs)
    model.load_best_checkpoint()
    save_model(os.path.join(model_dir, 'model.p'), model)
    metrics = model.evaluate(test_generator)

    with open(os.path.join(model_dir, 'metrics.json'), 'w') as file:
        json.dump(metrics, file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='original, cropped or segmented')
    parser.add_argument('--loader', help='resize or pad')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    sys.exit(main(args))
