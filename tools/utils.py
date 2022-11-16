import torch
import os
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from DeepDr import DeepDrVAN, DeepDrVBC, DeepDrABC, DeepDrRVO
from model import MaskedSwinTransformer as MST
from dataloader import Transform, RVO


def load_image(image, image_path):
    image_dir = os.path.join(image_path, image)

    return Image.open(image_dir).convert('RGB')


def get_deepdr(n, pretrain):
    deepdr = MST.swin_large_patch4_window12_384()
    deepdr.head = nn.Linear(in_features=1536, out_features=n, bias=True)
    pretrained_state_dict = torch.load(
        '../model/pretrained/' + pretrain + '.pth', map_location='cuda:0')
    state_dict = deepdr.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in state_dict}
    state_dict.update(pretrained_state_dict)
    deepdr.load_state_dict(state_dict)
    return deepdr


def get_dataloader(args):
    images = list(os.listdir(args.image_root_dir))
    data = RVO(images=images, loader=load_image, image_root_dir=args.image_root_dir, transform=Transform())
    data_iter = DataLoader(data, args.batch_size, shuffle=False, num_workers=8)
    return data_iter


def get_DeepDr(args):
    if args.module == 'DeepDrVAN':
        return DeepDrVAN.DeepDrVAN()
    if args.module == 'DeepDrVBC':
        return DeepDrVBC.DeepDrVBC()
    if args.module == 'DeepDrABC':
        return DeepDrABC.DeepDrABC()
    if args.module == 'DeepDrRVO':
        return DeepDrRVO.DeepDrRVO()
