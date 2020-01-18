import numpy as np
import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from utils import prepare_dataset
from mrcnn.utils import compute_ap


class PredictionConfig(Config):
    NAME = "trafficsign_cfg"
    NUM_CLASSES = 1 + 150
    # detect complains about `len(images) must be equal to BATCH_SIZE` without setting these
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def evaluate_model(dataset, model, cfg):
    APs = list()
    df = pd.DataFrame(columns=["image", "confidence",
                               "category", "xmin", "ymin", "xmax", "ymax"])
    i = 0
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=1)
        print(yhat)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(
            gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
        info = dataset.image_info[image_id]
        print(image_id, AP, info)
        for j in range(len(r["rois"])):
            # df.iloc[image_id] = [info["filename"], r["scores"][0]]
            print(j)
            row = {"image": info["filename"],
                   "confidence": r["scores"][j],
                   "category": r["class_ids"][j],
                   "xmin": r["rois"][j][0],
                   "ymin": r["rois"][j][1],
                   "xmax": r["rois"][j][2],
                   "ymax": r["rois"][j][3]}
            df = df.append(row, ignore_index=True)
            i += 1
        if i > 10:
            break
    # calculate the mean AP across all images
    df.to_csv("./results.csv")
    mAP = np.mean(APs)
    return mAP


data = utils.load("train.csv")
classes = utils.determine_classes(data)

train_set = prepare_dataset(data, classes)

cfg = PredictionConfig()
model = MaskRCNN(mode="inference", model_dir="./train", config=cfg)
model.load_weights(
    "./trafficsign_config20200116T0805/mask_rcnn_trafficsign_config_0005.h5", by_name=True)
train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
# test_mAP = evaluate_model(valid_set, model, cfg)
# print("Test mAP: %.3f" % test_mAP)
