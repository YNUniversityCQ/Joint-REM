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

    for build, mask, opt_mask, heatmap, target, img_name in test_dataloader:
        interation += 1

        print(interation)

        build, mask, opt_mask, heatmap, target = build.cuda(), mask.cuda(), opt_mask.cuda(), heatmap.cuda(), target.cuda()

        with torch.no_grad():
            pre, opt = model(build, mask, opt_mask, heatmap, target)

        # target
        test1 = torch.tensor([item.cpu().detach().numpy() for item in mask * target]).cuda()
        test1 = test1.squeeze(0)
        test1 = test1.squeeze(0)
        im = test1.cpu().numpy()
        image = im * 255
        images = Image.fromarray(image.astype(np.uint8))

        # 保存
        image_name = os.path.basename(img_name[0]).split('.')[0]
        images.save(os.path.join("image_result", f'{image_name}.png'))

if __name__ == '__main__':
 main_worker()