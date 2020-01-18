from mrcnn.config import Config
import numpy as np


class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "trafficsign_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 150
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class TrainingConfig(Config):
    NAME = "trafficsign_config"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 150
    STEPS_PER_EPOCH = 1000  # 5135 training images
    # MEAN_PIXEL = np.array([123.7, 116.8, 103.9]) ## default
    MEAN_PIXEL = np.array([130.17769851, 136.684966, 142.04479965])
