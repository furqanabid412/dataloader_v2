from parser_v2 import *

import yaml
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import timeit
import tqdm
from torch.utils.data import DataLoader
from modified_laser_net import Deep_Aggregation
import pytorch_model_summary



# data loader initialization

root='/home/share/dataset/semanticKITTI'
pc_root='E:/Datasets/SemanticKitti/dataset/Kitti'
laptop_root ='/media/furqan/Terabyte/Lab/datasets/semanticKitti'
DATA = yaml.safe_load(open('params/semantic-kitti.yaml', 'r'))
ARCH = yaml.safe_load(open('params/arch-params.yaml', 'r'))

train = 'train'
dataset=SemanticKitti(root=pc_root,sequences=DATA["split"]["train"],labels=DATA["labels"],
                      color_map=DATA["color_map"],learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                      sensor=ARCH["dataset"]["sensor"],multi_proj=ARCH["single"],max_points=ARCH["dataset"]["max_points"],train=train)


proj_multi_temporal_scan,proj_multi_temporal_label,scan_points,scan_range,scan_remission,scan_labels,proj_single_label,pixel_u,pixel_v = dataset[5]


lr = 0.002
num_classes = 21
model = Deep_Aggregation(5,[64,64,128],num_classes)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = nn.NLLLoss()




pytorch_model_summary(model, input_size=(5, 64, 1024))