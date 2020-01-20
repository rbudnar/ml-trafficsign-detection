import numpy as np
import pandas as pd
import utils
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from utils import prepare_dataset
from mrcnn.utils import compute_ap
import argparse


class PredictionConfig(Config):
    NAME = "trafficsign_cfg"
    NUM_CLASSES = 1 + 150
    # detect complains about `len(images) must be equal to BATCH_SIZE` without setting these
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def evaluate_model(dataset, model, cfg, classes):
    APs = list()
    df = pd.DataFrame(columns=["image", "confidence",
                               "category", "xmin", "ymin", "xmax", "ymax"])
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(
            gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        if AP is not None:
            APs.append(AP)
        info = dataset.image_info[image_id]
        print(image_id, AP)
        for j in range(len(r["rois"])):
            row = {"image": info["filename"],
                   "confidence": r["scores"][j],
                   "category": classes[r["class_ids"][j] + 1],
                   "xmin": r["rois"][j][0],
                   "ymin": r["rois"][j][1],
                   "xmax": r["rois"][j][2],
                   "ymax": r["rois"][j][3]}
            df = df.append(row, ignore_index=True)
    # calculate the mean AP across all images
    df.to_csv("./results.csv", index=False)
    mAP = np.mean(APs)
    return mAP


def run(train_csv, imagedir, modelpath):
    train_data = utils.load(train_csv)
    classes = utils.determine_classes(train_data)

    train_set = prepare_dataset(train_data, imagedir, classes)

    cfg = PredictionConfig()
    model = MaskRCNN(mode="inference", model_dir=imagedir, config=cfg)
    model.load_weights(modelpath, by_name=True)
    train_mAP = evaluate_model(train_set, model, cfg, classes)
    print("Train mAP: %.3f" % train_mAP)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--modelpath", help="path and filename for model file to load",
                    default="./trafficsign_config20200118T1111/mask_rcnn_trafficsign_config_0010.h5")
    ap.add_argument("-d", "--imagedir",
                    help="path to image directory.", default="./train")
    ap.add_argument(
        "-c", "--csv", help="path and filename to csv file from which the dataset will be built.", default="train.csv")

    args = vars(ap.parse_args())
    run(args["csv"], args["imagedir"], args["modelpath"])
