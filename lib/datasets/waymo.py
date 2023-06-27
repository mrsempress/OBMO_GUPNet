import os
import numpy as np
import cv2
from lib.datasets.kitti import KITTI
from PIL import Image

class Waymo(KITTI):
    def __init__(self, root_dir, split, cfg):
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        
        ##l,w,h
        self.cls_mean_size = np.array([[1.74902460, 0.85362147, 0.90802619],
                                       [1.79205100, 2.10208423, 4.79851555],
                                       [1.76897299 ,0.83370790, 1.76691953]])
        self.resolution = np.array([1056, 704])

        # data split loading
        assert split in ['train', 'val']
        self.split = split
        split_dir = os.path.join(root_dir, 'waymo', 'ImageSets', split + '_tiny.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = os.path.join(root_dir, 'waymo', 'validation' if split == 'val' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_0')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'filter_label_0')
        
        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'random_flip':0.0, 'random_crop':1.0, 'scale':0.4, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    # dataset = Waymo('../data/waymo', 'train', cfg)
    dataset = Waymo('/Volumes/Elements SE/tmp/', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, P2, coord_range, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()

        # test heatmap
        heatmap = targets['heatmap'][0]
        heatmap = np.uint8(np.floor(heatmap[0].numpy() * 255))  # cats id
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = Image.fromarray(heatmap)
        heatmap.show()
        break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
