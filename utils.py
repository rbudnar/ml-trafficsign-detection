import cv2
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir
import numpy as np
import ast
from TrafficSignDataset import TrafficSignDataset


def show_img(df, idx):
    row = df.iloc[idx]
    show_img_path(f'./train/{row["image"]}', row["object"])


def show_img_path(path, data):
    img = cv2.imread(path)
    fig, ax = plt.subplots(figsize=(30, 30))
    for obj in data:
        # draw_ellipse(img, obj)
        draw_rect(img, obj)
    ax.imshow(img, interpolation='nearest')


def draw_ellipse(img, data):
    if data is None:
        return
    e = data["ellipse"]
    # https://www.geeksforgeeks.org/python-opencv-cv2-ellipse-method/
    cv2.ellipse(img, (int(e[0][0]), int(e[0][1])), (int(
        e[1][0]), int(e[1][1])), int(e[2]), 0, 360, 255, 2)


def draw_rect(img, data):
    bbox = data["bbox"]
    startX = int(bbox["xmin"])
    startY = int(bbox["ymin"])
    endX = int(bbox["xmax"])
    endY = int(bbox["ymax"])
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)


def load(path, is_train=True):
    df = pd.read_csv(path, dtype={'object': object})
    if is_train:
        df["object"] = df["object"].apply(lambda x: ast.literal_eval(x))
        df["counts"] = df["object"].apply(lambda x: len(x))
    return df


def determine_classes(df):
    # expand rows from csv that may contain multiple bounding boxes into individual rows
    temp = pd.DataFrame(df["object"].explode())
    # grab the category from each BB
    temp["category"] = temp["object"].apply(lambda x: x.get("category"))
    return temp["category"].unique()


def prepare_dataset(df, image_dir, classes):
    dataset = TrafficSignDataset()
    dataset.load_dataset(image_dir, classes, df)
    dataset.prepare()
    return dataset


def generate_test_file(input_dir, output_filename):
    '''
    Generates and saves a csv file to setup the TrafficSignDataset to run predictions against. 
    This will not generate or store ground truth information in the file.

    input_dir: Expects a directory with images labeled as CAX_TestXXXX.jpg
    output_filename: Name of the file (no extension)

    Result: this saves the resulting dataframe to a csv file in the current directory.
    '''

    files = np.array(listdir(input_dir))
    df = pd.DataFrame(files, columns=["image"])
    df["id"] = df["image"].apply(lambda x: x.split(".")[0])
    df["id2"] = df["image"].apply(lambda t: int(
        t.split(".")[0].split("CAX_Test")[1]))
    df = df.sort_values("id2").drop(columns=["id2"])
    df.head()
    df.to_csv(f"{output_filename}.csv", index=False)
