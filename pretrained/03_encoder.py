"""
Image/SimCLR => Base Encoder = ResNet
EEG/Segmentation/SimCLR => Base Encoder = VGGNet
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle



class TupleDataset(Dataset):
    def __init__(self, tuple_data):
        self.features = tuple_data[0]  # 첫 번째 요소 (특성)
        self.labels = tuple_data[1]  # 두 번째 요소 (라벨)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class VGG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.enc5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()


    def forward(self, x):
        features = []

        x1 = self.enc1(x)
        features.append(x1)
        print(x1.shape)

        x2 = self.enc2(x1)
        features.append(x2)
        print(x2.shape)

        x3 = self.enc3(x2)
        features.append(x3)
        print(x3.shape)

        x4 = self.enc4(x3)
        features.append(x4)
        print(x4.shape)

        x5 = self.enc5(x4)
        features.append(x5)
        print(x5.shape)

        x6 = self.flatten(x5)
        print(x6.shape)

        return x6


class ResNet(nn.Module):
    pass

class EfficientNet(nn.Module):
    pass



# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#
# import pickle
# import copy
# with open('train_data.pkl', 'rb') as f:
#     train_data = pickle.load(f)
#
# train_data = copy.deepcopy(train_data)

# train_data = TupleDataset(train_data)
# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64, drop_last=True)
#
# train_features, train_labels = next(iter(train_dataloader))
# print(len(train_data))        # 170245
# print(len(train_dataloader))  # 2660
# print(f"Feature batch shape: {train_features.size()}")  # torch.Size([64, 4, 3000])
# print(f"Labels batch shape: {train_labels.size()}")  # torch.Size([64, 1, 3000])

# model = VGG(in_channels=4, num_classes=6)
# model.to(device)
#
# train_features = train_features.to(device)
# model_output = model(train_features)
# print(model_output)
# print(model_output.shape)


