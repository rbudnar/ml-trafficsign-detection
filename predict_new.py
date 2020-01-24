import numpy as np
from matplotlib import pyplot
import utils
from mrcnn.model import MaskRCNN
from configs import PredictionConfig
from mrcnn.model import mold_image
from utils import prepare_dataset
from matplotlib.patches import Rectangle
import argparse
import cv2


def plot(dataset, model, cfg, image_path, classes):
    image = cv2.imread(image_path)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = np.expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]
    pyplot.axis('off')
    pyplot.imshow(image)
    # get the context for drawing boxes
    pyplot.title('Predicted')
    ax2 = pyplot.gca()
    # plot each box
    for i in range(len(yhat["rois"])):
        class_name = classes[yhat["class_ids"][i] - 1]
        y1, x1, y2, x2 = yhat["rois"][i]
        add_box(x1, y1, x2, y2, ax2, class_name, "red")


def add_box(x1, y1, x2, y2, ax, class_name, color):
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color=color)
    # draw the box
    ax.add_patch(rect)
    ax.text(x1, y1, class_name, color="yellow", fontsize=13)


def run(csv, model_dir, model_file, img_name):
    data = utils.load(csv)
    classes = utils.determine_classes(data)
    train_set = prepare_dataset(data, model_dir, classes)

    cfg = PredictionConfig()
    model = MaskRCNN(mode="inference", model_dir=model_dir, config=cfg)
    model.load_weights(model_file, by_name=True)

    image_path = f"{model_dir}/{img_name}"
    plot(train_set, model, cfg, image_path, classes)
    pyplot.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--modelpath", help="path and filename for model file to load",
                    default="./trafficsign_config20200121T0807/mask_rcnn_trafficsign_config_0010.h5")
    ap.add_argument("-d", "--imagedir",
                    help="path to image directory.", default="./test")
    ap.add_argument(
        "-c", "--csv", help="path and filename to csv file from which the dataset will be built.", default="train.csv")
    ap.add_argument("-i", "--image", help="image to predict bounding boxes and classes from.",
                    default="CAX_Test1.jpg")
    args = vars(ap.parse_args())
    run(args["csv"], args["imagedir"], args["modelpath"], args["image"])
