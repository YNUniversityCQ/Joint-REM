import os
import torch
import numpy as np
from PIL import Image
from models import modules
from dataloaders import loaders1
from torch.utils.data import Dataset, DataLoader

# loading model
torch.cuda.set_device(0)
device = torch.device("cuda")
model = modules.jointnet1()
model.load_state_dict(torch.load('model_result/20/var=24/main_model.pt'))
model.to(device)

def main_worker():

    # loading test data
    test_data = loaders1.phase1(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)
    interation = 0
    for sample, build, target, img_name in test_dataloader:
        sample, build, target = sample.cuda(), build.cuda(), target.cuda()
        interation += 1

        with torch.no_grad():
            pre = model(sample, build)

        # target
        test1 = torch.tensor([item.cpu().detach().numpy() for item in sample]).cuda()
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

        # 保存
        image_name = os.path.basename(img_name[0]).split('.')[0]
        predict1.save(os.path.join("image_result", f'{image_name}.png'))
        # images.save(os.path.join("image_result", f'{image_name}.png'))

if __name__ == '__main__':
 main_worker()