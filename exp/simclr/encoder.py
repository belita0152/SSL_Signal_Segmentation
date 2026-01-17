"""
Image/SimCLR => Base Encoder = ResNet
EEG/Segmentation/SimCLR => Base Encoder = VGGNet
"""

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, in_channels):
        super(VGG, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv1d(32, 48, kernel_size=3),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.final_length = self.get_final_length(in_channels, 3000)

    def get_final_length(self, channel_size, input_size):
        x = torch.randn(1, channel_size, input_size)  # batch는 상관 없으므로 1로 고정
        x = self.forward(x)
        return x.view(-1).shape[0]  # 뒷부분이 1차원으로 합쳐짐. view(-1) -> 이 벡터의 크기가 projection head의 input_dim

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        return x


class ResNet(nn.Module):
    pass

class EfficientNet(nn.Module):
    pass
