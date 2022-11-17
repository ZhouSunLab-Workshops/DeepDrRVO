import torch
from torch import nn


class DeepDrVAN(nn.Module):
    def __init__(self, data_iter=None, get_deepdr=None
                 ):
        super(DeepDrVAN, self).__init__()

        self.deepdr = get_deepdr(3, 'DeepDrVAN').eval()
        self.data_iter = data_iter
        self.softmax = nn.Softmax(dim=1)

    def predict(self, x):
        y_ = self.deepdr(x)
        y_ = self.softmax(y_)
        y = y_.argmax(axis=1)
        return y

    def forward(self):
        results = {}
        for images, names in self.data_iter:
            images = images.to(torch.device('cuda:0'))
            labels = self.predict(images)
            labels = list(labels.cpu().detach().numpy())
            for (i, name) in enumerate(names):
                if labels[i] == 0:
                    tag = 'RvO'
                if labels[i] == 1:
                    tag = 'RaO'
                if labels[i] == 2:
                    tag = 'Normal'
                results[name] = tag
        return results
