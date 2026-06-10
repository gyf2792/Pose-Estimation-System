import torch
import torch.nn as nn
import timm


class MobilePose(nn.Module):
    def __init__(self, num_keypoints=4, pretrained=True):
        super().__init__()
        # 加载 MobileNetV3 Large，只取特征层
        # 输出 stride = 32 (即 512输入 -> 8x8输出),下采样
        self.backbone = timm.create_model('mobilenetv3_large_100',
                                          pretrained=pretrained,
                                          features_only=True)

        # 获取最后一层特征的通道数 (960)
        in_channels = self.backbone.feature_info[-1]['num_chs']

        # 上采样头：将 8x8 特征图 -> 64x64 热力图
        # 需要放大 8 倍 (Stride 8)
        self.head = nn.Sequential(
            # 第一次上采样 (x2)
            nn.ConvTranspose2d(in_channels, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第二次上采样 (x2)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 第三次上采样 (x2) -> 此时变为 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 最终输出层 (通道数=4)
            nn.Conv2d(64, num_keypoints, kernel_size=1, stride=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]  # 取最后一层特征
        out = self.head(x)
        return out