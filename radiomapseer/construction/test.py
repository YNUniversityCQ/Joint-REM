import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from models import modules
from dataloaders import loaders
# from torchmetrics import R2Score
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader


def ssim(img1, img2):
    # img1, img2: numpy数组，范围 0~255 或 0~1

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 高斯滤波参数
    sigma = 1.5
    win_size = 11

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # 高斯平滑
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 * img1, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

# loading model
torch.cuda.set_device(0)
device = torch.device("cuda")
model = modules.jointnet()
model.load_state_dict(torch.load('model_result/main_model.pt'))
model.to(device)

def main_worker():

    # loading test data
    test_data = loaders.phase2(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=0)

    interation = 0
    err1 = []
    err2 = []
    err3 = []
    err4 = []
    err5 = []

    for build, mask, opt_mask, heatmap, target, img_name in test_dataloader:
        interation += 1

        build, mask, opt_mask, heatmap, target = build.cuda(), mask.cuda(), opt_mask.cuda(), heatmap.cuda(), target.cuda()

        with torch.no_grad():
            pre = model(build, mask, opt_mask, heatmap, target)

        # target
        test1 = torch.tensor([item.cpu().detach().numpy() for item in target]).cuda()
        test1 = test1.squeeze(0)
        test1 = test1.squeeze(0)
        im = test1.cpu().numpy()
        image = im * 255
        images = Image.fromarray(image.astype(np.uint8))

        # predict
        test = torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
        test = test.squeeze(0)
        test = test.squeeze(0)
        im1 = test.cpu().numpy()
        predict = im1 * 255
        predict1 = Image.fromarray(predict.astype(np.uint8))

        # calculate rmse
        rmse1 = np.sqrt(np.mean((im - im1) ** 2))
        err1.append(rmse1)
        # calculate nmse
        nmse1 = np.mean((im - im1) ** 2)/np.mean((0 - im) ** 2)
        err2.append(nmse1)
        # calculate mae
        mae1 = np.mean(np.abs(im - im1))
        err3.append(mae1)
        # calculate psnr
        mse = np.mean((image - predict) ** 2)
        psnr1 = 10 * np.log10((255 ** 2) / mse)
        err4.append(psnr1)
        # calculate ssim
        ssim1 = ssim(image, predict)
        err5.append(ssim1)

        # 保存
        image_name = os.path.basename(img_name[0]).split('.')[0]
        images.save(os.path.join("image_result", f'{image_name}_target.png'))
        predict1.save(os.path.join("image_result", f'{image_name}_predict1.png'))
        # print(f'saving to {os.path.join("image_result", image_name)}', "RMSE:", rmse1, "NMSE:", nmse1)
        print(interation)

        # the number total of 8000
        if interation >= 8000:
            break
    rmse_err = sum(err1)/len(err1)
    nmse_err = sum(err2) / len(err2)
    mae_err = sum(err3) / len(err3)
    psnr_err = sum(err4) / len(err4)
    ssim_err = sum(err5) / len(err5)

    print('测试集均方根误差：', rmse_err)
    print('测试集归一化均方误差：', nmse_err)
    print('测试集平均绝对误差：', mae_err)
    print('测试集信噪比峰值：', psnr_err)
    print('测试集结构相似度：', ssim_err)

if __name__ == '__main__':
 main_worker()