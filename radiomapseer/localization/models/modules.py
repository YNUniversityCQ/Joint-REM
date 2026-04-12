import torch
from torch import nn
from thop import profile

def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    )

def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )

# U-Net
class jointnet1(nn.Module):

    def __init__(self, inputs=2):
        super().__init__()

        # 编码器：共9层，6层下采样
        self.conv0 = convrelu(inputs, 6, 3, 1, 1)
        self.conv1 = convrelu(6, 40, 5, 2, 2)
        self.conv2 = convrelu(40, 50, 5, 2, 2)
        self.conv3 = convrelu(50, 60, 5, 2, 1)
        self.conv4 = convrelu(60, 100, 5, 2, 2)
        self.conv5 = convrelu(100, 100, 3, 1, 1)
        self.conv6 = convrelu(100, 150, 5, 2, 2)
        self.conv7 = convrelu(150, 300, 5, 2, 2)
        self.conv8 = convrelu(300, 500, 5, 2, 2)
        #
        # 解码器：共9层，6层上采样
        self.up_conv1 = convreluT(500, 300, 4, 1)
        self.up_conv2 = convreluT(300+300, 150, 4, 1)
        self.up_conv3 = convreluT(150+150, 100, 4, 1)
        self.conv30 = convrelu(100+100, 100, 3, 1, 1)    # 不变大小，只改通道
        self.up_conv4 = convreluT(100+100, 60, 6, 2)
        self.conv40 = convrelu(60+60, 50, 5, 2, 1)       # 不变大小，只改通道
        self.up_conv5 = convreluT(50+50, 40, 6, 2)
        self.up_conv6 = convreluT(40+40, 6, 6, 2)
        self.conv60 = convrelu(6+6, 1, 5, 2, 1)             # 不变大小，只改通道

    def forward(self,  sample, build):

        x = torch.cat([sample, build], dim=1)

        # 编码器
        layer0 = self.conv0(x)
        layer1 = self.conv1(layer0)
        layer2 = self.conv2(layer1)
        layer3 = self.conv3(layer2)
        layer4 = self.conv4(layer3)
        layer5 = self.conv5(layer4)
        layer6 = self.conv6(layer5)
        layer7 = self.conv7(layer6)
        layer8 = self.conv8(layer7)

        # 解码器
        up_layer1 = self.up_conv1(layer8)
        cat1 = torch.cat([up_layer1, layer7], dim=1)
        up_layer2 = self.up_conv2(cat1)
        cat2 = torch.cat([up_layer2, layer6], dim=1)
        up_layer3 = self.up_conv3(cat2)
        cat3 = torch.cat([up_layer3, layer5], dim=1)
        layer30 = self.conv30(cat3)
        cat4 = torch.cat([layer30, layer4], dim=1)
        up_layer4 = self.up_conv4(cat4)
        cat5 = torch.cat([up_layer4, layer3], dim=1)
        layer40 = self.conv40(cat5)
        cat6 = torch.cat([layer40, layer2], dim=1)
        up_layer5 = self.up_conv5(cat6)
        cat7 = torch.cat([up_layer5, layer1], dim=1)
        up_layer6 = self.up_conv6(cat7)
        cat8 = torch.cat([up_layer6, layer0], dim=1)
        layer60 = self.conv60(cat8)
        output = layer60

        return output

def test():
    x = torch.randn((1, 2, 256, 256)).cuda()

    print('==> Building model..')

    model = jointnet1()
    model.cuda()

    flops, params = profile(model, (x, ))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()