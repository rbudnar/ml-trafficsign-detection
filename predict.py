import numpy as np
from matplotlib import pyplot
import utils
from mrcnn.model import MaskRCNN
from configs import PredictionConfig
from mrcnn.model import mold_image
from utils import prepare_dataset
from matplotlib.patches import Rectangle
import argparse
# plot a number of photos with ground truth and predictions


def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        plot(dataset, model, cfg, i, i, n_images)
    # show the figure
    pyplot.show()


def plot(dataset, model, cfg, i, img_id, n_images=1):
    print(f"loading img_id {img_id}")
    print(dataset.image_info[img_id])
    image = dataset.load_image(img_id)
    mask, _ = dataset.load_mask(img_id)

    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = np.expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]
    # define subplot
    pyplot.subplot(n_images, 2, i * 2 + 1)
    # turn off axis labels
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(image)
    if i == 0:
        pyplot.title('Actual')
        # plot masks

    ax = pyplot.gca()
    boxes, width, height = dataset.extract_boxes(dataset.image_info[img_id])
    for box in boxes:
        x1, y1, x2, y2, cat = box
        add_box(x1, y1, x2, y2, ax, "blue")

    # get the context for drawing boxes
    pyplot.subplot(n_images, 2, i * 2 + 2)
    # turn off axis labels
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(image)
    if i == 0:
        pyplot.title('Predicted')
    ax2 = pyplot.gca()
    # plot each box
    for box in yhat['rois']:
        y1, x1, y2, x2 = box
        add_box(x1, y1, x2, y2, ax2, "red")


def add_box(x1, y1, x2, y2, ax, color):
    # x1, y1, x2, y2 = box
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color=color)
    # draw the box
    ax.add_patch(rect)


def run(csv, model_dir, model_file, img_name):
    data = utils.load(csv)
    classes = utils.determine_classes(data)
    train_set = prepare_dataset(data, classes)

    cfg = PredictionConfig()
    model = MaskRCNN(mode="inference", model_dir=model_dir, config=cfg)
    model.load_weights(model_file, by_name=True)

    idx = data[data["image"] == img_name].index[0]
    plot(train_set, model, cfg, 0, idx)
    pyplot.show()
    # plot_actual_vs_predicted(train_set, model, cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--modelpath", help="path and filename for model file to load",
                    default="./trafficsign_config20200116T0805/mask_rcnn_trafficsign_config_0005.h5")
    ap.add_argument("-d", "--imagedir",
                    help="path to image directory.", default="./train")
    ap.add_argument(
        "-c", "--csv", help="path and filename to csv file from which the dataset will be built.", default="train.csv")
    ap.add_argument("-i", "--image", help="image to predict bounding boxes and classes from.",
                    default="CAX_Train28.jpg")
    args = vars(ap.parse_args())
    print(args)
    run(args["csv"], args["imagedir"], args["modelpath"], args["image"])
