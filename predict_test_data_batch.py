import numpy as np
import pandas as pd
import utils
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from utils import prepare_dataset
from configs import PredictionConfig
import argparse


def evaluate_model(dataset, model, cfg, classes):
    df = pd.DataFrame(columns=["image", "confidence",
                               "category", "xmin", "ymin", "xmax", "ymax", "id"])
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        # image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(
        #     dataset, cfg, image_id, use_mini_mask=False)
        image = dataset.load_image(image_id)
        # mask, _ = dataset.load_mask(image_id)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]

        info = dataset.image_info[image_id]
        id = int(info["filename"].split(".")[0].split("CAX_Test")[1])
        print(image_id, info, id)
        for j in range(len(r["rois"])):
            row = {"image": info["filename"],
                   "confidence": r["scores"][j],
                   "category": classes[r["class_ids"][j] - 1],
                   "xmin": r["rois"][j][0],
                   "ymin": r["rois"][j][1],
                   "xmax": r["rois"][j][2],
                   "ymax": r["rois"][j][3],
                   "id": id}
            df = df.append(row, ignore_index=True)

    df = df.sort_values("id").drop(columns=["id"])
    df.to_csv("./submission_results.csv", index=False)


def run(train_csv, test_csv, imagedir, model_path):
    data = utils.load(train_csv)
    classes = utils.determine_classes(data)

    test_data = utils.load(test_csv, is_train=False)
    test_set = prepare_dataset(test_data, imagedir, classes)

    cfg = PredictionConfig()
    model = MaskRCNN(mode="inference", model_dir=imagedir, config=cfg)
    model.load_weights(model_path, by_name=True)
    evaluate_model(test_set, model, cfg, classes)
    print("Done!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--modelpath", help="path and filename for model file to load",
                    default="./trafficsign_config20200118T1111/mask_rcnn_trafficsign_config_0010.h5")
    ap.add_argument("-d", "--imagedir",
                    help="path to image directory.", default="./test")
    ap.add_argument(
        "-c", "--csv", help="path and filename to training csv file from which the class names will be built.", default="train.csv")
    ap.add_argument(
        "-t", "--test_csv", help="path and filename to csv file from which the dataset will be built.", default="test.csv")

    args = vars(ap.parse_args())
    run(args["csv"], args["test_csv"], args["imagedir"], args["modelpath"])
