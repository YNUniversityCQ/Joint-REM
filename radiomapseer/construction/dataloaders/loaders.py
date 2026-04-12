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

class phase2(Dataset):

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/",
                 numTx=80,
                 sample_nums=5,
                 pick_times=1,
                 var=24,
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
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
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = self.data + "image/"
        self.build = self.data + "build/"
        self.antenna = self.data + "antenna/"
        self.heatmap = self.data + "heatmap/var=24/15/"
        self.opt_position = self.data + "position_r/15/"

    def __len__(self):
        return (self.num2 - self.num1) * self.numTx

    def __getitem__(self, idx):
        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr + self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        builds = os.path.join(self.build, name1)
        build = np.asarray(io.imread(builds))

        antennas = os.path.join(self.antenna, name2)
        arr_antenna = np.asarray(io.imread(antennas))

        target = os.path.join(self.simulation, name2)
        target = np.asarray(io.imread(target))

        # 采样
        nonzero_coords = list(zip(*np.nonzero(target)))

        sample = random.sample(nonzero_coords, self.sample_nums)

        imRx = np.zeros((self.width, self.height))

        for m in range(self.sample_nums):
            single_Rx = sample[m]
            imRx[single_Rx[0], single_Rx[1]] = 1

        sample = imRx

        # # 位置热图
        # antenna = np.unravel_index(np.argmax(arr_antenna), arr_antenna.shape)

        # heatmap = GaussianHeatMap(self.height, self.width, antenna[1], antenna[0], self.var)

        heatmap = os.path.join(self.heatmap, name2)
        heatmap = np.asarray(io.imread(heatmap))

        opt_mask = os.path.join(self.opt_position, name2)
        opt_mask = np.asarray(io.imread(opt_mask))

        # To tensor
        build = self.transform(build/255).type(torch.float32)
        mask = self.transform(sample).type(torch.float32)
        opt_mask = self.transform(opt_mask/255).type(torch.float32)
        heatmap = self.transform(heatmap/255).type(torch.float32)
        target = self.transform(target/255).type(torch.float32)

        return build, mask, opt_mask, heatmap, target, name2


def test():
    dataset = phase2(phase='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for x, y, z in loader:
        print(x.shape, y.shape, z)

if __name__ == "__main__":
    test()

