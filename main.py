import argparse
from tools.utils import get_dataloader, get_DeepDr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module',
                        default='DeepDrRVO')
    parser.add_argument('--image_root_dir',
                        default='')

    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    data = get_dataloader(args)
    DeepDr = get_DeepDr(args)
    results = DeepDr(data)
