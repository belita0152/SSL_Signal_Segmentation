import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimCLR(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim=16, temperature: float=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, False),
        )
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    def forward(self, h1, h2):
        # print(h1.shape, h2.shape)
        z1, z2 = self.projection_head(h1, h2)
        device = z1.device
        loss = self.infonce_loss(z1, z2).to(device)
        return loss

    # pass through projection head
    def projection_head(self, h1, h2):
        b = h1.shape[0]
        h1 = h1.view(b, -1)  # (torch) .view  ==  (numpy) .reshape
        h2 = h2.view(b, -1)
        # print(b, h1.shape, h2.shape)  # 64, torch.Size([64, 96000], torch.Size([64, 96000]
        # learnable nonlinear transformation between representation and contrastive loss
        z_i = self.projector(h1)
        z_j = self.projector(h2)
        return z_i, z_j

    def infonce_loss(self, z_i, z_j):
        # print(z_i.shape, z_j.shape)  # torch.Size([128, 16]), torch.Size([128, 16])
        # InfoNCE loss function (contrastive loss)
        device = z_i.device
        b = z_i.shape[0]  # ******* 64 = batch
        n = 2 * b  # 128

        # labels
        class_labels = torch.cat([torch.arange(b) for i in range(2)], dim=0)
        class_labels = (class_labels.unsqueeze(0) == class_labels.unsqueeze(1)).float()  # torch.Size([128, 128]
        class_labels = class_labels.to(device)

        # features
        features = torch.cat((z_i, z_j), dim=0)
        features = F.normalize(features, dim=-1)  # torch.Size([128, 16])
        similarity_matrix = torch.matmul(features, features.T) / self.temperature # torch.Size([128, 128])

        mask = torch.eye(class_labels.shape[0], dtype=torch.bool).to(device)  # 대각선이 1인 diag matrix [128, 128]

        class_labels = class_labels[~mask].view(class_labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        assert class_labels.shape == similarity_matrix.shape  # [128, 127]로 동일

        positives = similarity_matrix[class_labels.bool()].view(class_labels.shape[0], -1)  # [128, 1]
        negatives = similarity_matrix[~class_labels.bool()].view(similarity_matrix.shape[0], -1)  # [128, 126]

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss = self.criterion(logits, labels)
        loss = loss / n  # 각 batch 당 loss

        return loss
