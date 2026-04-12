import torch
import einops
import torch.nn as nn
from thop import profile
from einops import rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .model import BasicBlock, Bottleneck
from .attention import SpatialAttention, ChannelAttention, PixelAttention
from .attention import spatial_attention, channel_attention, cbam_block

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

def gumbel_topk(logits, k=5, tau=0.1, eps=1e-10):
    """
    logits: [B, 1, H, W]
    返回 soft one-hot mask，可微化训练
    """
    B, _, H, W = logits.shape
    flat = logits.view(B, -1)

    # 添加 Gumbel 噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(flat) + eps) + eps)
    y = (flat + gumbel_noise) / tau

    # 连续 top-k 近似，使用 sparsemax-like trick
    topk_vals, topk_idx = torch.topk(y, k, dim=1)
    mask = torch.zeros_like(flat)
    mask.scatter_(1, topk_idx, 1.0)

    # 使用 straight-through trick 保持梯度
    y_soft = F.softmax(y, dim=1)
    mask = mask + y_soft - y_soft.detach()  # ST Gumbel-Softmax

    return mask.view(B, 1, H, W)

class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        # dilation rate
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        # 标准卷积(3*3)
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]

        out = torch.cat(out, 1)

        # 加融合机制需要开
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)

        return x * (1 - mask) + out * mask

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

# joint attention
class joint_attention(nn.Module):
    def __init__(self, dim, kernel, reduction=8):
        super(joint_attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel, stride=1, padding=1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        res = self.act(res)
        return res
        
# sensor U-Net
class sensor(nn.Module):

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

    def forward(self,  x):

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

# tx U-Net
class transmitter(nn.Module):

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

    def forward(self,  x):

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
        
class jointnet(nn.Module):
    def __init__(self, inputs=3):
        super().__init__()

        # dense down sampling
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)
        self.pool_8 = nn.MaxPool2d(8, 8)

        # dense up sampling
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # encoder
        self.conv0 = convrelu(inputs, 64, 5, 2, 2)
        self.conv1 = convrelu(64 + inputs, 128, 5, 2, 2)
        self.conv2 = convrelu(128 + 64 + inputs, 256, 5, 2, 2)
        self.conv3 = convrelu(256 + 128 + 64 + inputs, 512, 5, 2, 2)

        # middle
        self.AOT0 = nn.Sequential(*[AOTBlock(64, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT1 = nn.Sequential(*[AOTBlock(128, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT2 = nn.Sequential(*[AOTBlock(256, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT3 = nn.Sequential(*[AOTBlock(512, [1, 2, 4, 8]) for _ in range(1)])

        # ours
        self.att0 = joint_attention(64, 3)
        self.att1 = joint_attention(128, 3)
        self.att2 = joint_attention(256, 3)
        self.att3 = joint_attention(512, 3)

        self.conv11 = convrelu(1024, 256, 3, 1, 1)
        self.conv22 = convrelu(1024, 128, 3, 1, 1)
        self.conv33 = convrelu(1024, 64, 3, 1, 1)

        # decoder
        self.up_conv0 = convreluT(512, 256, 4, 1)
        self.up_conv1 = convreluT(256, 128, 4, 1)
        self.up_conv2 = convreluT(128, 64, 4, 1)
        self.up_conv3 = convreluT(64, 1, 4, 1)
        
        # # 实例化传感布局网络
        # self.sensor = sensor()
        # # 实例化源定位网络
        # self.transmitter = transmitter()

    def forward(self, build, mask, opt_mask, heatmap, target):

        # sensor_net_out = self.sensor(torch.cat([build, mask * target], 1))
        #
        # opt_mask = gumbel_topk(sensor_net_out, k=5, tau=0.1)
        #
        # tx_net_out = self.transmitter(torch.cat([build, mask * target], dim=1))
        #
        # # KL散度
        # opt_prob = F.softmax(opt_mask.view(opt_mask.size(0), -1), dim=1)
        # best_prob = F.softmax(best_mask.view(best_mask.size(0), -1), dim=1)
        # l1 = F.kl_div(opt_prob.log(), best_prob, reduction='batchmean')
        #
        # # MSE
        # l2 = F.mse_loss(tx_net_out, heatmap, reduction='mean')

        inputs = torch.cat([build, opt_mask * target, heatmap], dim=1)

        # stage 1
        layer0 = self.conv0(inputs)
        layer0 = self.AOT0(layer0)
        layer0 = self.att0(layer0)

        # stage 2
        down1_1 = self.pool_2(inputs)
        cat1 = torch.cat([down1_1, layer0], dim=1)
        layer1 = self.conv1(cat1)
        layer1 = self.AOT1(layer1)
        layer1 = self.att1(layer1)

        # stage 3
        down1_2 = self.pool_4(inputs)
        down2_1 = self.pool_2(layer0)
        cat2 = torch.cat([down1_2, down2_1, layer1], dim=1)
        layer2 = self.conv2(cat2)
        layer2 = self.AOT2(layer2)
        layer2 = self.att2(layer2)

        # stage 4
        down1_3 = self.pool_8(inputs)
        down2_2 = self.pool_4(layer0)
        down3_1 = self.pool_2(layer1)
        cat3 = torch.cat([down1_3, down2_2, down3_1, layer2], dim=1)
        layer3 = self.conv3(cat3)
        layer3 = self.AOT3(layer3)
        layer3 = self.att3(layer3)

        # stage 1
        up_layer0 = self.up_conv0(layer3)

        # stage 2
        up1_1 = self.up_2(layer3)
        up_cat1 = torch.cat([up1_1, up_layer0, layer2], dim=1)
        up_layer1 = self.conv11(up_cat1)
        up_layer1 = self.up_conv1(up_layer1)

        # stage 3
        up1_2 = self.up_4(layer3)
        up2_1 = self.up_2(up_layer0)
        up_cat2 = torch.cat([up1_2, up2_1, up_layer1, layer1], dim=1)
        up_layer2 = self.conv22(up_cat2)
        up_layer2 = self.up_conv2(up_layer2)

        # stage 4
        up1_3 = self.up_8(layer3)
        up2_2 = self.up_4(up_layer0)
        up3_1 = self.up_2(up_layer1)
        up_cat2 = torch.cat([up1_3, up2_2, up3_1, up_layer2, layer0], dim=1)
        up_layer3 = self.conv33(up_cat2)
        up_layer3 = self.up_conv3(up_layer3)

        return up_layer3

# Debug
def test():
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 1, 256, 256))
    z = torch.randn((1, 1, 256, 256))
    w = torch.randn((1, 1, 256, 256))
    t = torch.randn((1, 1, 256, 256))

    model = jointnet()

    flops, params = profile(model, (x, y, z, w, t))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

    model.cuda()
    preds = model(x.cuda(), y.cuda(), z.cuda(), w.cuda(), t.cuda())
    print(preds.shape)


if __name__ == "__main__":
    test()