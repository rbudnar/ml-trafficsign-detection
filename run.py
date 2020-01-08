import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils
import json
import ast
from TrafficSignDataset import TrafficSignDataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from sklearn.model_selection import train_test_split

# train_set = TrafficSignDataset()
# train_set.load_dataset('.', "train.csv", is_train=True)
# train_set.prepare()
# print('Train: %d' % len(train_set.image_ids))

# image_id = 0
# image = train_set.load_image(image_id)
# print(image.shape)
# # load image mask
# mask, class_ids = train_set.load_mask(image_id)
# # print(mask.shape)
# # plot image
# # pyplot.imshow(image)
# # # plot mask
# # pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
# # pyplot.show()

# # for i in range(9):
# #     # define subplot
# #     pyplot.subplot(330 + 1 + i)
# #     # turn off axis labels
# #     pyplot.axis('off')
# #     # plot raw pixel data
# #     image = train_set.load_image(i)
# #     pyplot.imshow(image)
# #     # plot all masks
# #     mask, _ = train_set.load_mask(i)
# #     for j in range(mask.shape[2]):
# #         pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
# # # show the figure
# # pyplot.show()

# bbox = extract_bboxes(mask)
# # display image with masks and bounding boxes
# display_instances(image, bbox, mask, class_ids, train_set.class_names)


def run(path_to_csv="train.csv"):
    data = utils.load(path_to_csv)
    classes = utils.determine_classes(data)

    train, test = train_test_split(data, test_size=0.2)
    train_set = prepare_dataset(train, classes)
    valid_set = prepare_dataset(test, classes, is_train=False)

    print('Train: %d' % len(train_set.image_ids))
    print('Test: %d' % len(valid_set.image_ids))


def prepare_dataset(df, classes, is_train=True):
    dataset = TrafficSignDataset()
    dataset.load_dataset(".", df, classes, is_train)
    dataset.prepare()
    return dataset


run()
