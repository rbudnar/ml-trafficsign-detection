import utils
from sklearn.model_selection import train_test_split
from mrcnn.model import MaskRCNN
from utils import prepare_dataset
from configs import TrainingConfig

import importlib
wandb = importlib.util.find_spec("wandb")
wandb_found = wandb is not None

if wandb_found:
    import wandb
    from wandb.keras import WandbCallback


def run(path_to_csv="train.csv"):
    wandb.init()
    data = utils.load(path_to_csv)
    classes = utils.determine_classes(data)

    train, test = train_test_split(data, test_size=0.2)
    train_set = prepare_dataset(train, "./train/", classes)
    valid_set = prepare_dataset(test, "./train/", classes)

    print('Train: %d' % len(train_set.image_ids))
    print('Test: %d' % len(valid_set.image_ids))

    config = TrainingConfig()
    callbacks = []
    if wandb_found:
        callbacks.append(WandbCallback())
        config.STEPS_PER_EPOCH = wandb.config.STEPS_PER_EPOCH
        config.LEARNING_RATE = wandb.config.LEARNING_RATE
        config.LEARNING_MOMENTUM = wandb.config.LEARNING_MOMENTUM
        config.WEIGHT_DECAY = wandb.config.WEIGHT_DECAY
    else:
        # configure params through directly editing the TrainingConfig class
        pass
    model = MaskRCNN(mode="training", model_dir="train", config=config)
    model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=[
                       "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    # tb = keras.callbacks.TensorBoard(log_dir="./logs")
    model.train(train_set, valid_set,
                learning_rate=config.LEARNING_RATE, epochs=wandb.config.EPOCHS, layers="heads", custom_callbacks=callbacks)


run()
