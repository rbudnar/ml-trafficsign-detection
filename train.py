import utils
from sklearn.model_selection import train_test_split
from mrcnn.model import MaskRCNN
from utils import prepare_dataset
import keras
from configs import TrainingConfig


def run(path_to_csv="train.csv"):
    data = utils.load(path_to_csv)
    classes = utils.determine_classes(data)

    train, test = train_test_split(data, test_size=0.2)
    train_set = prepare_dataset(train, classes)
    valid_set = prepare_dataset(test, classes)

    print('Train: %d' % len(train_set.image_ids))
    print('Test: %d' % len(valid_set.image_ids))

    config = TrainingConfig()
    model = MaskRCNN(mode="training", model_dir="train", config=config)
    model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=[
                       "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    tb = keras.callbacks.TensorBoard(log_dir="./logs")
    model.train(train_set, valid_set,
                learning_rate=config.LEARNING_RATE, epochs=5, layers="heads", custom_callbacks=[tb])


run()
