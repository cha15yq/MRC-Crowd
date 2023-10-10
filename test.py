import argparse

import numpy as np

from datasets.dataset import Crowd
import torch
import os
from torch.utils.data import DataLoader
from model.model import vgg19
import torch.nn.functional as F


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downsample-ratio', default=8, type=int,
                        help='the downsample ratio of the model')
    parser.add_argument('--data-dir', default='QNRF/val',
                        help='the directory of the data')
    parser.add_argument('--model-path', default='best_model.pth',
                        help='the path to the model')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='the number of samples in a batch')
    parser.add_argument('--device', default='0',
                        help="assign device")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    dataset = Crowd(args.data_dir, 512, args.downsample_ratio, method='val')
    dataloader = DataLoader(dataset, 1, shuffle=False, pin_memory=False)
    model = vgg19(25)
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, device))
    model.eval()
    file = open('result.txt', 'w')
    step = 0
    epoch_res = []
    for im, gt_counts, name in dataloader:
        inputs = im.to(device)
        with torch.set_grad_enabled(False):
            B, C, H, W = inputs.size()
            output = model(inputs)[0]
            res = gt_counts[0].item() - torch.sum(output).item()
            epoch_res.append(res)
            print('{}/{} {}: predict:{}, gt:{}, diff:{}'.format(step, len(dataset), name, torch.sum(output).item(), gt_counts[0].item(), res),  np.abs(np.array(epoch_res)).mean())
            step += 1
            file.write('{}/{} {} {}\n'.format(name, len(dataset), torch.sum(output).item(), gt_counts[0].item(), res))
    epoch_res = np.abs(np.array(epoch_res))
    print('MAE:{:.2f}, MSE:{:.2f}'.format(epoch_res.mean(), np.sqrt((epoch_res**2).mean())))
