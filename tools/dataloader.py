import torch
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Dataset

from ietk import methods



class Transform:
    def __init__(self):
        self.transform1 = transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(384),

        ])

        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y_ = self.transform1(x)
        y_ = np.array(y_) / 255
        y_ = methods.brighten_darken(y_, 'A+C+X+Z')
        y = self.transform2(y_)
        y = y.type(torch.FloatTensor)
        return y


class RVO(Dataset):
    def __init__(self, images=None, loader=None, image_root_dir=None, transform=None,
                 ):
        super(RVO, self).__init__()

        self.images = images
        self.loader = loader
        self.image_root_dir = image_root_dir
        self.transform = transform

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        label = image
        image_dir = self.image_root_dir
        image = self.loader(image, image_dir)
        image = self.transform(image)
        return image, label


