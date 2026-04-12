from __future__ import print_function, division
import os
import random
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings

import matplotlib.pyplot as plt

# 消除运行中的红色字体
warnings.filterwarnings("ignore")

# 定义高斯热图
def GaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    # 生成网格坐标
    x = np.arange(img_width)
    y = np.arange(img_height)
    xx, yy = np.meshgrid(x, y)

    # 计算距离平方并生成高斯
    dist_sq = (xx - c_x) ** 2 + (yy - c_y) ** 2
    gaussian_map = np.exp(-dist_sq / (2 * variance * variance))
    return gaussian_map

# 定义位置挑选
def select_Rx_positions(Rx, num_Rx, interations):
    selections = []
    for _ in range(interations):
        selected_indices = np.random.choice(len(Rx), num_Rx, replace=False)
        selected_positions = Rx[selected_indices]
        selections.append(selected_positions)
    return selections

class phase1(Dataset):

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/",
                 numTx=80,
                 sample_nums=20,
                 pick_times=1,
                 var=24,                            # 越大, 范围越大
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 701, 1, dtype=np.int16)
            np.random.seed(2025)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        self.data = data
        self.numTx = numTx
        self.transform = transform
        self.height = 256
        self.width = 256
        self.var = self.height / 256 * var
        self.sample_nums = sample_nums
        self.pick_times = pick_times
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 700
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 0
            self.num2 = 700

        self.RSS_simulation = self.data + "image/"
        self.build = self.data + "build/"
        self.antenna = self.data + "antenna/"

    def __len__(self):
        return (self.num2 - self.num1 + 1) * self.numTx * self.pick_times

    def __getitem__(self, idx):
        numMapPhase = self.num2 - self.num1 + 1
        idxMap, idxTx, idxpick = np.unravel_index(idx, (numMapPhase, self.numTx, self.pick_times))

        dataset_map = self.maps[idxMap + self.num1]

        # build maps name
        name1 = str(dataset_map) + ".png"

        # Radio maps name
        name2 = str(dataset_map) + "_" + str(idxTx) + ".png"

        # loading build maps
        build_path = os.path.join(self.build, name1)
        build_arr = np.asarray(io.imread(build_path))

        # loading antenna maps
        antenna_path = os.path.join(self.antenna, name2)
        antenna_arr = np.asarray(io.imread(antenna_path))

        # loading Radio maps
        RSS_path = os.path.join(self.RSS_simulation, name2)
        RSS_arr = np.asarray(io.imread(RSS_path))

        nonzero_coords = list(zip(*np.nonzero(RSS_arr)))

        sample = random.sample(nonzero_coords, self.sample_nums)

        imRx = np.zeros((self.width, self.height))

        for m in range(self.sample_nums):
            single_Rx = sample[m]
            imRx[single_Rx[0], single_Rx[1]] = 1

        sample_arr_RSS = imRx * RSS_arr

        # argmax target position
        antenna = np.unravel_index(np.argmax(antenna_arr), antenna_arr.shape)

        target = GaussianHeatMap(self.height, self.width, antenna[1], antenna[0], self.var)

        # transfer tensor
        samples = self.transform(sample_arr_RSS).type(torch.float32)
        builds = self.transform(build_arr).type(torch.float32)
        target = self.transform(target).type(torch.float32)

        loc_r = torch.from_numpy(np.asarray(antenna[0], dtype=np.float32))
        loc_c = torch.from_numpy(np.asarray(antenna[1], dtype=np.float32))
        loc = torch.stack((loc_r, loc_c), dim=0)

        return samples,  builds, target, loc, name2

def test():
    dataset = phase1(phase='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for w, x, y, z in loader:
        print(w.shape, x.shape, y.shape, z)

if __name__ == "__main__":
    test()

