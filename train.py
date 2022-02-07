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


# stats=torch.cuda.memory_stats("cuda")
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

num_classes = len(DATA["learning_map_inv"])

# weighted loss
wl=weighted_loss(num_classes,DATA,ARCH)
loss_weights=wl.get_weights(0.01)

# proj_multi_temporal_scan,proj_multi_temporal_label,scan_points,scan_range,scan_remission,scan_labels,proj_single_label,pixel_u,pixel_v = dataset[5]


lr = 0.002
model = Deep_Aggregation(5,[64,64,128],num_classes).cuda()
criterion = nn.NLLLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

dataloader=DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4,pin_memory=True, drop_last=True)
wandb.init(project='single_laser',entity='furqanabid' ,resume='allow')

evaluator = iouEval(num_classes,"cuda",[])


# proj_multi_temporal_scan=torch.squeeze(proj_multi_temporal_scan,0)
# model summary
# summary(model,proj_multi_temporal_scan)

# data visualization
# visualize = visualizer(DATA["color_map"],"magma",DATA["learning_map_inv"])
# label = visualize.map(proj_single_label, DATA["learning_map_inv"])
# input_tensor=proj_multi_temporal_scan[0,1]
# visualize.range_image_2D(input_tensor.numpy(),"range projection",10,"magma")
# visualize.label_image_2D(label,"Label",10)

step_losses = []
epoch_losses = []

epochs = 25
for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, (proj_single_scan,proj_single_label) in enumerate(dataloader):

        proj_single_scan = proj_single_scan.cuda()
        proj_single_label = proj_single_label.cuda().long()

        class_probs = model.forward(proj_single_scan)
        optimizer.zero_grad()
        loss = criterion(F.log_softmax(class_probs, dim=1), proj_single_label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())

        with torch.no_grad():
            evaluator.reset()
            argmax = class_probs.argmax(dim=1)
            # argmax_numpy=argmax.cpu().numpy()
            evaluator.addBatch(argmax, proj_single_label)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()


        print("Batch :", batch_idx ,"| epoch :", epoch, "| step loss :",loss.item(),"| accuracy :",accuracy.item(),"| iou :",jaccard.item())
        wandb.log({"loss": loss.item(),"batch":batch_idx,"accuracy":accuracy.item(),"iou":jaccard.item()})

    epoch_losses.append(epoch_loss / len(dataloader))
    print("********epoch loss********",epoch_losses)

torch.save(model.state_dict(),"model1.pth")

print("debugging")