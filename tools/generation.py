import argparse
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
from model.FewSampleGenerater import GI, GIHR
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', default=100)
    parser.add_argument('--upscale_factor', default=4)
    parser.add_argument('--args.GI_path', default='../model/pretrained/GI_')
    parser.add_argument('--args.GIHR_path', default='../model/pretrained/GIHR_')
    parser.add_argument('--args.RaO', default='BRAO')
    parser.add_argument('--save_path', default='../Synthetic_CFPs/')
    args = parser.parse_args()
    return args


def HR_cfp_generation(args):
    noise = torch.randn(args.num, 100, 1, 1, device='cuda:0')
    GI_ = GI().eval().to('cuda:0')
    GI_.load_state_dict(torch.load(args.GI_path + args.RaO + '.pth', map_location=lambda storage, loc: storage))
    GIHR_ = GIHR(args.upscale_factor).eval().to('cuda:0')
    GIHR_.load_state_dict(torch.load(args.GIHR_path + args.RaO + '.pth', map_location=lambda storage, loc: storage))
    xI = GI_(noise)
    xI = ToPILImage()(xI[0].data)
    for i in range(args.num):
        xI = vutils.make_grid(xI[i], padding=2, normalize=True)
        xI = ToPILImage()(xI.data)
        xI = Variable(ToTensor()(xI), volatile=False).unsqueeze(0)
        XIHR = GIHR_(XIHR)
        XIHR.save(args.save_path + args.RaO + '_HR_CFP{}.png'.format(str(i)))


if __name__ == "__main__":
    args = get_args()
    HR_cfp_generation(args)
