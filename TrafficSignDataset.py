from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from os import listdir
import utils


class TrafficSignDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, df, classes, is_train=True):
        self.df = df
        #self.df = utils.load(path_to_csv)
        images_dir = dataset_dir + ("/train/" if is_train else "/test/")

        # classes = utils.determine_classes(self.df)
        for i in range(classes.shape[0]):
            self.add_class("dataset", i+1, classes[i])

        for (index, data) in self.df.iterrows():
            filename = data["image"]
            img_path = images_dir + filename
            # add to dataset
            self.add_image('dataset', image_id=index,
                           path=img_path, filename=filename)

    # load all bounding boxes for an image
    def extract_boxes(self, img_info):
        # load and parse the file
        id = img_info["id"]
        object_info = self.df.loc[id]["object"]
        boxes = list()

        # extract image dimensions
        for data in object_info:
            bbox = data["bbox"]
            startX = int(bbox["xmin"])
            startY = int(bbox["ymin"])
            endX = int(bbox["xmax"])
            endY = int(bbox["ymax"])
            coors = [startX, startY, endX, endY, data["category"]]
            boxes.append(coors)
            print(coors)
        width = 2048
        height = 2048
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        print(info)
        boxes, w, h = self.extract_boxes(info)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(box[4]))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class TrafficSignConfig(Config):
    NAME = "trafficsign_config"
    NUM_CLASSES = 1 + 150
    STEPS_PER_EPOCH = 200
