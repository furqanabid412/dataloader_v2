from parser_v2 import *

import yaml
import torch
import os
import numpy as np
import timeit
from torch.utils.data import DataLoader



root='/home/share/dataset/semanticKITTI'
pc_root='E:/Datasets/SemanticKitti/dataset/Kitti'
# laptop_root ='/media/furqan/Terabyte/Lab/datasets/semanticKitti'
DATA = yaml.safe_load(open('params/semantic-kitti.yaml', 'r'))
ARCH = yaml.safe_load(open('params/arch-params.yaml', 'r'))


train = 'train'
train_dataset=SemanticKitti(root=pc_root,sequences=DATA["split"]["train"],labels=DATA["labels"],
                            color_map=DATA["color_map"],learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],multi_proj=ARCH["multi"],max_points=ARCH["dataset"]["max_points"],train=train)

dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=True, drop_last=True)

time_taken = timeit.default_timer()

overall_mean=0
overall_std=0
data_size = 0

feature_means = np.array([0, 0, 0, 0, 0])
feature_std = np.array([0, 0, 0, 0, 0])

for batch_idx, (proj_multi_temporal_scan,proj_multi_temporal_label,scan_points,scan_range,
                scan_remission,scan_labels,proj_single_label,pixel_u,pixel_v) in enumerate (dataloader):
    # print ("batch ",batch_idx,"time taken",timeit.default_timer() - time_taken)

    proj_multi_temporal_scan=proj_multi_temporal_scan.numpy()

    bs,ts,mr,ft,h,w=np.shape(proj_multi_temporal_scan)

    # calculate mean and s.d for each range
    for features in range(ft):
        array=proj_multi_temporal_scan[0,0,0,features,:,:]
        feature_means[features]+=np.mean(array)
        feature_std[features]+=np.std(array)
        print("mean :", feature_means[features], "std :", feature_std[features], "batch_idx : ", batch_idx ,"feature",features)


    data_size+=1
    # input=proj_multi_temporal_scan.cuda()
    # labels=proj_multi_temporal_label.cuda(non_blocking=True).long()
    # time_taken =timeit.default_timer()


overall_mean=feature_means/data_size
overall_std=feature_std/data_size

print("prog")
