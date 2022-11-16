import torch
from torch import nn
from tools.utils import get_deepdr


class DeepDrABC(nn.Module):
    def __init__(self, data_iter=None
                 ):
        super(DeepDrABC, self).__init__()
        self.deepdr = get_deepdr(2, 'DeepDrABC').eval()
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
                if labels[i] == 1:
                    tag = 'BRAO'
                else:
                    tag = 'CRAO'
                results[name] = tag
        return results
