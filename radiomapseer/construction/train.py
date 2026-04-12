from __future__ import print_function, division
import os
import time
import copy
import torch
import warnings
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchsummary import summary
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from dataloaders import loaders
from models import modules

if __name__ == '__main__':
    print('GPU:', torch.cuda.device_count())
    device = torch.device("cuda:0")

    # ignore warning
    warnings.filterwarnings("ignore")

    # speed
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    Radio_train = loaders.phase2(phase="train")
    Radio_val = loaders.phase2(phase="val")

    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    main_dataloaders = {
        'train': DataLoader(Radio_train, batch_size=8, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(Radio_val, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    }

    # loading network parameter
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.backends.cudnn.enabled
    model = modules.jointnet()

    # def load_weights(model_path):
    #     pretrained = torch.load(model_path)
    #     model.load_state_dict(pretrained, strict=True)
    #     del pretrained
    #
    # # loading pre-train model parameter
    # model_path = "model_result/main_model.pt"
    #
    # load_weights(model_path)

    model.cuda()

    def combine_loss(pred, target, metrics):
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        metrics['MSE_loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def print_metrics(metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))

    def train_model(model, optimizer, scheduler, num_epochs=1, targetType="dense"):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            since = time.time()

            # 训练并验证
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("learning rate", param_group['lr'])

                    model.train()
                else:
                    model.eval()

                metrics = defaultdict(float)
                epoch_samples = 0
                if targetType == "dense":
                    for build, mask, opt_mask, heatmap, target, name in tqdm(main_dataloaders[phase]):

                        build, mask, opt_mask, heatmap, target = build.cuda(), mask.cuda(), opt_mask.cuda(), heatmap.cuda(), target.cuda()

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):

                            outputs = model(build, mask, opt_mask, heatmap, target)

                            loss = combine_loss(outputs, target, metrics)

                            # print(loss)

                            if phase == 'train':
                                loss.backward(torch.ones_like(loss))
                                optimizer.step()

                        epoch_samples += target.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['MSE_loss'] / epoch_samples

                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        # saving best weight
        model.load_state_dict(best_model_wts)
        return model

    # conduct training
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    model = train_model(model, optimizer_ft, exp_lr_scheduler)

    # build the dir
    try:
        os.mkdir('model_result')
    except OSError as error:
        print(error)

    # saving
    torch.save(model.state_dict(), 'model_result/main_model.pt')