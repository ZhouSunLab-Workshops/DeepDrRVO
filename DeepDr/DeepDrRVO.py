import torch
from torch import nn
from tools.utils import get_deepdr


class DeepDrRVO(nn.Module):
    def __init__(self, data_iter=None
                 ):
        super(DeepDrRVO, self).__init__()

        self.DeepDrVAN = get_deepdr(3, 'DeepDrVAN').eval()
        self.DeepDrVBC = get_deepdr(2, 'DeepDrVBC').eval()
        self.DeepDrABC = get_deepdr(2, 'DeepDrABC').eval()
        self.data_iter = data_iter
        self.softmax = nn.Softmax(dim=1)

    def predict(self, x, names):
        y_van = self.DeepDrVAN(x)
        y_van = self.softmax(y_van).cpu().detach().numpy()
        y_van = y_van.argmax(axis=1)
        x_RvO_idx = [i for i in range(len(y_van)) if y_van[i] == 0]
        x_RaO_idx = [i for i in range(len(y_van)) if y_van[i] == 1]
        x_normal_idx = [i for i in range(len(y_van)) if y_van[i] == 2]
        x_RvO = torch.index_select(x, 0, torch.tensor(x_RvO_idx))
        x_RaO = torch.index_select(x, 0, torch.tensor(x_RaO_idx))
        names_RvO = names[x_RvO_idx]
        names_RaO = names[x_RaO_idx]
        names_normal = names[x_normal_idx]
        y_RvO = self.DeepDrVBC(x_RvO)
        y_RvO = self.softmax(y_RvO).cpu().detach().numpy()
        y_RvO = y_RvO.argmax(axis=1)
        y_RaO = self.DeepDrABC(x_RaO)
        y_RaO = self.softmax(y_RaO).cpu().detach().numpy()
        y_RaO = y_RaO.argmax(axis=1)
        y_normal = y_van[x_normal_idx]

        return (y_RvO, y_RaO, y_normal), (names_RvO, names_RaO, names_normal)

    def forward(self):
        results = {}
        for images, names in self.data_iter:
            images = images.to(torch.device('cuda:0'))
            labels, names = self.predict(images, names)
            y_RvO, y_RaO, y_normal = labels
            names_RvO, names_RaO, names_normal = names
            for (i, name) in enumerate(names_RvO):
                if y_RvO[i] == 0:
                    tag = 'CRVO'
                if y_RvO[i] == 1:
                    tag = 'BRVO'
                results[name] = tag
            for (i, name) in enumerate(names_RaO):
                if y_RaO[i] == 0:
                    tag = 'CRAO'
                if y_RaO[i] == 1:
                    tag = 'BRAO'
                results[name] = tag
            for (i, name) in enumerate(names_normal):
                results[name] = 'Normal'

        return results
