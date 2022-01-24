import os
import numpy as np
import torch
from torch.utils.data import Dataset
from laserscan_v2 import *


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               multi_proj,        # multi projection parameters
               max_points=150000,   # max number of points present in dataset
               train='train',
               ):
    # copying the params to self instance

    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]

    self.intervals = multi_proj["intervals"]
    # self.intervals = []
    self.timeframe = multi_proj["timeframes"]
    self.do_calib =multi_proj["calibrate"]
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.train = train

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)

    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    # placeholder for calibration
    self.calibrations = []
    self.times = []
    self.poses = []


    self.frames_in_a_seq=[]
    frames_in_a_seq = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      # print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      # sort for correspondance
      scan_files.sort()
      # append list
      self.scan_files.append(scan_files)

      # check all scans have labels
      if self.train == 'train':
        label_path = os.path.join(self.root, seq, "labels")
        label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_path)) for f in fn if is_label(f)]
        label_files.sort()
        # check all scans have labels
        assert (len(scan_files) == len(label_files))
        self.label_files.append(label_files)


      if self.do_calib :
        self.calibrations.append(self.parse_calibration(os.path.join(self.root, seq, "calib.txt")))  # read caliberation
        self.times.append(np.loadtxt(os.path.join(self.root, seq, 'times.txt'), dtype=np.float32))  # read times
        poses_f64 = self.parse_poses(os.path.join(self.root, seq, 'poses.txt'), self.calibrations[-1])
        self.poses.append([pose.astype(np.float32) for pose in poses_f64])  # read poses


      frames_in_a_seq.append(len(scan_files))



    self.frames_in_a_seq = np.array(frames_in_a_seq).cumsum()

    self.scan = LaserScan(interv=self.intervals, ch=5, H=self.sensor_img_H, W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up, fov_down=self.sensor_fov_down, tr=self.train, proj=True,
                          multi_proj=True)

    # print("Using {} scans from sequences {}".format(len(self.scan_files),
    #                                                 self.sequences))


  def parse_calibration(self, filename):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
      key, content = line.strip().split(":")
      values = [float(v) for v in content.strip().split()]
      pose = np.zeros((4, 4))
      pose[0, 0:4] = values[0:4]
      pose[1, 0:4] = values[4:8]
      pose[2, 0:4] = values[8:12]
      pose[3, 3] = 1.0
      calib[key] = pose
    calib_file.close()
    return calib

  def parse_poses(self, filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
      values = [float(v) for v in line.strip().split()]
      pose = np.zeros((4, 4))
      pose[0, 0:4] = values[0:4]
      pose[1, 0:4] = values[4:8]
      pose[2, 0:4] = values[8:12]
      pose[3, 3] = 1.0
      poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses

  def get_seq_and_frame(self, index):
    # function takes index and convert it to seq and frame number

    if index < self.frames_in_a_seq[0]:
      return 0, index

    else:
      seq_count = len(self.frames_in_a_seq)
      for i in range(seq_count):
        fr = index + 1
        if i < seq_count - 1 and self.frames_in_a_seq[i] < fr and self.frames_in_a_seq[i + 1] > fr:
          # print("here")
          return i + 1, index - self.frames_in_a_seq[i]

        elif i < seq_count - 1 and self.frames_in_a_seq[i] == fr:
          return i, index - self.frames_in_a_seq[i - 1]

        elif i < seq_count - 1 and fr == self.frames_in_a_seq[-1]:
          return seq_count - 1, index - self.frames_in_a_seq[-2]


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

  def __len__(self):
    return self.frames_in_a_seq[-1]


  def __getitem__(self, index):

    seq,frame = self.get_seq_and_frame(index)

    scan_paths=[]
    label_paths=[]

    # get the list of filenames (scan and labels)
    # for multiple time frames

    pose0 = self.poses[seq][frame]

    proj_multi_temporal_scan = []
    proj_multi_temporal_label = []


    for timeframe in range(self.timeframe):
      if frame - timeframe >= 0:
        curr_frame = frame - timeframe
      else:
        curr_frame = 0
      temp_scan_path = self.scan_files[seq][curr_frame]
      if self.train == 'train' :
        temp_label_path = self.label_files[seq][curr_frame]
      else:
        temp_label_path =[]

      curr_pose = self.poses[seq][curr_frame]

      # check whether a coordinate transformation is needed or not

      if timeframe == 0 or np.array_equal(pose0, curr_pose):
        ego_motion = False
      else:
        ego_motion = True

      # opening the scan(and label) file
      self.scan.open_scan(temp_scan_path,temp_label_path, pose0, curr_pose, ego_motion=ego_motion)

      multi_proj_scan=np.copy(self.scan.concated_proj_range)
      proj_multi_temporal_scan.append(multi_proj_scan)

      if self.train == 'train':
        multi_proj_label=np.copy(self.scan.concated_proj_semlabel)
        proj_multi_temporal_label.append(multi_proj_label)

      if timeframe == 0 :
        # other params for post processing - only for frame "t"
        original_points = np.copy(self.scan.point)

        total_points = original_points.shape[0]

        scan_points = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        scan_points[:total_points] = torch.from_numpy(original_points)

        scan_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        scan_range[:total_points] = torch.from_numpy(np.copy(self.scan.unproj_range))

        scan_remission = torch.full([self.max_points], -1.0, dtype=torch.float)
        scan_remission[:total_points] = torch.from_numpy(np.copy(self.scan.remission))

        if self.train == 'train':
          # mapping classes and saving as a tensor
          original_labels = self.map(np.copy(self.scan.label),self.learning_map)

          scan_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
          scan_labels[:total_points] = torch.from_numpy(np.squeeze(original_labels,axis=1))

          proj_single_label = self.map(np.copy(self.scan.proj_sem_label), self.learning_map)
          proj_single_label = torch.tensor(proj_single_label)

        else:
          scan_labels = torch.tensor([])
          proj_single_label = torch.tensor([[]])

        pixel_u = torch.full([self.max_points], -1, dtype=torch.long)
        pixel_u[:total_points] = torch.from_numpy(np.copy(self.scan.proj_x))

        pixel_v = torch.full([self.max_points], -1, dtype=torch.long)
        pixel_v[:total_points] = torch.from_numpy(np.copy(self.scan.proj_y))



    proj_multi_temporal_scan=torch.tensor(np.copy(proj_multi_temporal_scan))

    if self.train =='train':
      proj_multi_temporal_label = self.map(np.copy(proj_multi_temporal_label),self.learning_map)
      proj_multi_temporal_label = torch.tensor(proj_multi_temporal_label)
    else:
      proj_multi_temporal_label = torch.tensor([])



    return proj_multi_temporal_scan,proj_multi_temporal_label,scan_points,scan_range,scan_remission,\
           scan_labels,proj_single_label,pixel_u,pixel_v


