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
from pytorch_model_summary import summary
from visualizer_v2 import visualizer
from iou import iouEval
import wandb
from loss_w import weighted_loss
# torch.cuda.empty_cache()



root='/home/share/dataset/semanticKITTI'
pc_root='E:/Datasets/SemanticKitti/dataset/Kitti'
laptop_root ='/media/furqan/Terabyte/Lab/datasets/semanticKitti'
DATA = yaml.safe_load(open('params/semantic-kitti.yaml', 'r'))
ARCH = yaml.safe_load(open('params/arch-params.yaml', 'r'))

train = 'train'
dataset=SemanticKitti(root=pc_root,sequences=DATA["split"]["train"],labels=DATA["labels"],
                      color_map=DATA["color_map"],learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                      sensor=ARCH["dataset"]["sensor"],multi_proj=ARCH["multi"],max_points=ARCH["dataset"]["max_points"],train=train)

dataset.check_ego_motion(15,2)

print("test")