import numpy as np
import pandas as pd
import utils
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from utils import prepare_dataset
from configs import PredictionConfig
import argparse


def evaluate_model(dataset, model, cfg, classes, model_path):
    df = pd.DataFrame(columns=["image", "confidence",
                               "category", "xmin", "ymin", "xmax", "ymax", "id"])
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image = dataset.load_image(image_id)
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
        for (idx, box) in enumerate(r['rois']):
            class_name = classes[r["class_ids"][idx] - 1]
            y1, x1, y2, x2 = box
            row = {"image": info["filename"],
                   "confidence": r["scores"][idx],
                   "category": class_name,
                   "xmin": x1,
                   "ymin": y1,
                   "xmax": x2,
                   "ymax": y2}
            print(row)
            df = df.append(row, ignore_index=True)

    df = df.sort_values("id").drop(columns=["id"])
    filesuffix = model_path.replace("/", "__")
    df.to_csv(f"./submission_results__{filesuffix}.csv", index=False)


def run(train_csv, test_csv, imagedir, model_path):
    data = utils.load(train_csv)
    classes = utils.determine_classes(data)

    test_data = utils.load(test_csv, is_train=False)
    test_set = prepare_dataset(test_data, imagedir, classes)

    cfg = PredictionConfig()
    model = MaskRCNN(mode="inference", model_dir=imagedir, config=cfg)
    model.load_weights(model_path, by_name=True)
    evaluate_model(test_set, model, cfg, classes, model_path)
    print("Done!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--modelpath", help="path and filename for model file to load",
                    default="./trafficsign_config20200121T0807/mask_rcnn_trafficsign_config_0010.h5")
    ap.add_argument("-d", "--imagedir",
                    help="path to image directory.", default="./test")
    ap.add_argument(
        "-c", "--csv", help="path and filename to training csv file from which the class names will be built.", default="train.csv")
    ap.add_argument(
        "-t", "--test_csv", help="path and filename to csv file from which the dataset will be built.", default="test.csv")

    args = vars(ap.parse_args())
    run(args["csv"], args["test_csv"], args["imagedir"], args["modelpath"])
