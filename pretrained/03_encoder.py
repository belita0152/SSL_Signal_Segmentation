"""
Image/SimCLR => Base Encoder = ResNet
EEG/Segmentation/SimCLR => Base Encoder = VGGNet
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle


# Tuple (train_data) -> x_train, y_train으로 분리
class TupleDataset(Dataset):  # 일종의 get data
    def __init__(self, tuple_data):
        self.features = tuple_data[0]  # x data
        self.labels = tuple_data[1]  # y labels

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
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        # x4 = self.enc4(x3)

        return x3


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
#
# train_data = TupleDataset(train_data)
# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64, drop_last=True)
# for data in train_dataloader:
#     print(data)



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


# train_features, train_labels = next(iter(train_dataloader))
# train_features = train_features.to(device)
# model_output = encoder(train_features)

"""
torch.Size([64, 64, 1500])
torch.Size([64, 128, 750])
torch.Size([64, 256, 375])
torch.Size([64, 512, 187])
torch.Size([64, 512, 93])
torch.Size([64, 47616])
"""
