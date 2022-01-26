import matplotlib.pyplot as plt
import numpy as np
from parser_v2 import *
import yaml
import torch
import os
import timeit
import tqdm
from torch.utils.data import DataLoader
import open3d

class visualizer():
    def __init__(self,label_colormap,image_colormap,learning_map_inv):
        self.label_colormap = label_colormap
        self.image_colormap = image_colormap
        self.learning_map_inv = learning_map_inv

    def label_image_2D(self, input, title, scale, show_plot=True, save_image=False, path=''):
        H,W=np.shape(input)
        plt.figure(figsize=(int(W / scale) - 4, int(H / scale) + 2))
        plt.title(title,fontsize=scale*5)
        image = np.array([[self.label_colormap[val] for val in row] for row in input], dtype='B')
        plt.imshow(image)
        plt.show()

    # 2D visualization any image

    def range_image_2D(self, input, title, scale, colormap="magma"):
        cmap = plt.cm.get_cmap(colormap, 10)
        H, W = np.shape(input)
        plt.figure(figsize=(int(W / scale) - 4, int(H / scale) + 2))
        plt.title(title)
        plt.imshow(input, cmap=cmap)
        plt.colorbar()
        plt.show()

    def pcl_3d(self,scan_points,scan_labels):

        pcd = open3d.geometry.PointCloud()
        scan_points = scan_points.numpy()
        scan_points = scan_points[scan_labels != -1, :]
        pcd.points = open3d.utility.Vector3dVector(scan_points)
        scan_labels = scan_labels.numpy()
        scan_labels = scan_labels[scan_labels != -1]
        scan_labels = self.map(scan_labels, self.learning_map_inv)
        colors = np.array([self.label_colormap[x] for x in scan_labels])
        pcd.colors = open3d.utility.Vector3dVector(colors / 255.0)
        vis = open3d.visualization.VisualizerWithKeyCallback()
        # vis.create_window(width=width, height=height, left=100)
        # vis.add_geometry(pcd)
        vis = open3d.visualization.draw_geometries([pcd])
        open3d.visualization.ViewControl()
    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

'''
root='/home/share/dataset/semanticKITTI'
pc_root='E:/Datasets/SemanticKitti/dataset/Kitti'
laptop_root ='/media/furqan/Terabyte/Lab/datasets/semanticKitti'
DATA = yaml.safe_load(open('params/semantic-kitti.yaml', 'r'))
ARCH = yaml.safe_load(open('params/arch-params.yaml', 'r'))


train = 'train'
dataset=SemanticKitti(root=pc_root,sequences=DATA["split"]["train"],labels=DATA["labels"],
                            color_map=DATA["color_map"],learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],multi_proj=ARCH["multi"],max_points=ARCH["dataset"]["max_points"],train=train)


proj_multi_temporal_scan,proj_multi_temporal_label,scan_points,scan_range,scan_remission,scan_labels,proj_single_label,pixel_u,pixel_v = dataset[5]

visualize = visualizer(DATA["color_map"],"magma",DATA["learning_map_inv"])


t,r,_,_=proj_multi_temporal_label.size()

# for timeframe in range(t):
timeframe=0
for multirange in range(r):
    label = proj_multi_temporal_label[timeframe][multirange]
    label = visualize.map(label, DATA["learning_map_inv"])
    title = "time frame :"+str(timeframe)+" range :"+str(multirange)
    visualize.label_image_2D(label, title, 10, show_plot=True, save_image=False, path='')


proj_single_label = visualize.map(proj_single_label, DATA["learning_map_inv"])
visualize.label_image_2D(proj_single_label, "original inage", 10, show_plot=True, save_image=False, path='')

visualize.pcl_3d(scan_points,scan_labels)



timeframe=0
multirange=0
r1=proj_multi_temporal_scan[timeframe][multirange][1]
m1=r1!=-1

multirange=1
r2=proj_multi_temporal_scan[timeframe][multirange][1]
m2=r2!=-1

multirange=2
r3=proj_multi_temporal_scan[timeframe][multirange][1]
m3=r3!=-1

multirange=3
r4=proj_multi_temporal_scan[timeframe][multirange][1]
m4=r4!=-1

# plt.imshow(m1)
# plt.show()
# plt.imshow(m2)
# plt.show()

m5=m2*m1
sum1=m5.sum()
m6=m2*m3
sum2=m6.sum()
m7=m3*m4
sum3=m7.sum()

plt.imshow(m5)
plt.show()
plt.imshow(m6)
plt.show()
plt.imshow(m7)
plt.show()

print("just testing")

'''
