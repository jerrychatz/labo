import pickle

import numpy as np
from PIL import Image


class Config:
    dataset_path = "/research/cbim/vast/gc745/research/LaBo/datasets/CIFAR10/cifar-10-batches-py/"
    output_path = "/research/cbim/vast/gc745/research/LaBo/datasets/CIFAR10/images/"

def pickle_load(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f, encoding='bytes')

def reshape(array):
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(array[:1024], (32, 32))  # Red channel
    image[...,1] = np.reshape(array[1024:2048], (32, 32))  # Green channel
    image[...,2] = np.reshape(array[2048:], (32, 32))  # Blue channel
    return image

def process_batches(data, class2images, mode):
    for batch in data:
        for i in range(len(batch[b'filenames'])):
            file_name = batch[b'filenames'][i].decode("utf-8")
            label = batch[b'labels'][i]
            array = batch[b'data'][i]
            class_name = all_class[label]
            image = Image.fromarray(reshape(array))
            image.save(f"{Config.output_path}{mode}/{file_name}")
            class2images[class_name].append(file_name)

if __name__ == '__main__':
    train_batches = [f"data_batch_{i}" for i in range(1, 6)]
    test_batch = "test_batch"
    meta_data = "batches.meta"

    train_data = [pickle_load(Config.dataset_path + batch) for batch in train_batches]
    test_data = pickle_load(Config.dataset_path + test_batch)
    meta = pickle_load(Config.dataset_path + meta_data)

    all_class = [label_name.decode("utf-8").replace("_", " ") for label_name in meta[b'label_names']]

    class2images_train = {c: [] for c in all_class}
    process_batches(train_data, class2images_train, "train")

    class2images_test = {c: [] for c in all_class}
    process_batches([test_data], class2images_test, "test")

    print(len(class2images_train['truck']), len(class2images_test['truck']))
