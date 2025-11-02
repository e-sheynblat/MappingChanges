import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Siamese(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Siamese, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # remove last FC
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(resnet.fc.in_features, 1)  # output probability

    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        diff = torch.abs(out1 - out2)
        out = torch.sigmoid(self.fc(diff))
        return out
