import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ast


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


def load(path):
    df = pd.read_csv(path, dtype={'object': object})
    df["object"] = df["object"].apply(lambda x: ast.literal_eval(x))
    df["counts"] = df["object"].apply(lambda x: len(x))
    return df


def determine_classes(df):
    # expand rows from csv that may contain multiple bounding boxes into individual rows
    temp = pd.DataFrame(df["object"].explode())
    # grab the category from each BB
    temp["category"] = temp["object"].apply(lambda x: x.get("category"))
    return temp["category"].unique()
