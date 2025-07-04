import torch.nn as nn
import torch
from collections import OrderedDict

class EnsembleModel(nn.Module):
    def __init__(
        self,
        densenet_hippo, 
        classifier_hippo_volumes,
        hidden_size=768 + 128,
        n_classes=2,
    ):
        super(EnsembleModel, self).__init__()

        self.densenet_hippo = densenet_hippo
        self.classifier_hippo_volumes = classifier_hippo_volumes

        self.n_classes = n_classes

        self.classifier_head =  nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_size, hidden_size // 2)),
            ('relu_fc1', nn.LeakyReLU()),
            ('dropout', nn.Dropout(0.3)),
            ('fc2', nn.Linear(hidden_size // 2, n_classes)),
            ('softmax', nn.Softmax(dim=1))
        ]))
        

    def forward(self, x):
        x1 = self.densenet_hippo(x['image'])
        x2 = self.classifier_hippo_volumes(x['data'])
                    
        y = self.classifier_head(torch.cat([x1, x2], dim=1))

        return y
    

class EnsembleModel(nn.Module):
    def __init__(
        self,
        densenet_hippo_left, 
        densenet_hippo_right,
        classifier_hippo_volumes,
        hidden_size=768 + 768 + 128,
        n_classes=2,
    ):
        super(EnsembleModel, self).__init__()

        self.densenet_hippo_left = densenet_hippo_left
        self.densenet_hippo_right = densenet_hippo_right
        self.classifier_hippo_volumes = classifier_hippo_volumes

        self.n_classes = n_classes

        self.classifier_head =  nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_size, hidden_size // 2)),
            ('relu_fc1', nn.LeakyReLU()),
            ('dropout', nn.Dropout(0.3)),
            ('fc2', nn.Linear(hidden_size // 2, n_classes)),
            ('softmax', nn.Softmax(dim=1))
        ]))
        

    def forward(self, x):
        x1 = self.densenet_hippo_left(x['left'])
        x2 = self.densenet_hippo_right(x['right'])
        x3 = self.classifier_hippo_volumes(x['data'])
                    
        y = self.classifier_head(torch.cat([x1, x2, x3], dim=1))

        return y