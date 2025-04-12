"""
Image/SimCLR => No Decoder
EEG/Segmentation/SimCLR => UNet-based Decoder
"""

import torch
import torch.nn as nn
from pretrained.encoder import VGG
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def get_pretrained_model(load_path=None):
    model = VGG(in_channels=4, num_classes=6)

    if load_path is not None:
        ckpt = torch.load(load_path, map_location=device)  # encoder only
        model.load_state_dict(ckpt, strict=True)

    for param in model.parameters():
        param.requires_grad = False

    return model

class UNet(nn.Module):
    def __init__(self, num_classes, encoder_ckpt):
        super(UNet, self).__init__()
        self.n_classes = num_classes
        self.backbone = get_pretrained_model(load_path=encoder_ckpt)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 48, kernel_size=3, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Sequential(
            nn.Conv1d(48, 32, kernel_size=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.outc = nn.Conv1d(32, 6, kernel_size=1)

        self.final_length = self.get_final_length(4, 3000)

    def get_final_length(self, channel_size, input_size):
        x = torch.randn(1, channel_size, input_size)
        x = self.forward(x)
        return x.view(-1).shape[0]

    def forward(self, x):
        # x = x.to(device)
        x1 = self.backbone.enc1(x)  # torch.Size([1, 48, 749])
        x2 = self.backbone.enc2(x1)  # torch.Size([1, 128, 186])
        x3 = self.backbone.enc3(x2)  # torch.Size([1, 256, 46])

        x = self.up1(x3)  # torch.Size([1, 256, 92])
        x = torch.cat([x, x3], dim=-1)  # torch.Size([1, 256, 138])
        x = self.conv1(x)  # torch.Size([1, 128, 136])

        x = self.up2(x)  # torch.Size([1, 128, 272])
        x = torch.cat([x, x2], dim=-1)  # torch.Size([1, 128, 458])
        x = self.conv2(x) # torch.Size([1, 64, 456])

        x = self.up3(x)  # torch.Size([1, 64, 912]))
        x = torch.cat([x, x1], dim=-1)  # torch.Size([1, 48, 1657])
        x = self.conv3(x)  # torch.Size([1, 32, 1655])

        logits = self.outc(x)
        return logits
