import argparse
import torch
import os
from utils.regression_trainer import Reg_Trainer


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', default="test", type=str,
                        help='what is it?')
    parser.add_argument('--label-info', default='label_list/ucf-40.txt', type=str,
                        help='the path to the label information')
    parser.add_argument('--seed', default=15, type=int, help='if not using seed, please set as -1')
    parser.add_argument('--crop-size', default=512, type=int,
                        help='the cropped size of the training data')
    parser.add_argument('--downsample-ratio', default=8, type=int,
                        help='the downsample ratio of the model')
    parser.add_argument('--data-dir', default='QNRF',
                        help='the directory of the data')

    parser.add_argument('--save-dir', default='history',
                        help='the directory for saving models and training logs')

    parser.add_argument('--max-num', default=2, type=int,
                        help='the maximum number of saved models ')
    parser.add_argument('--resume', default="",
                        help='the path of the resume training model')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='the number of samples in a batch')
    parser.add_argument('--num-labeled', default=4, type=int,
                        help='the number of labeled samples in a batch')

    # Optimizer
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='the learning rate')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='the number of workers')

    parser.add_argument('--device', default='0',
                        help="assign device")

    parser.add_argument('--start-epoch', default=0, type=int,
                        help='the number of starting epoch')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='the maximum number of training epoch')
    parser.add_argument('--start-val', default=300, type=int,
                        help='the starting epoch for validation')
    parser.add_argument('--val-epoch', default=1, type=int,
                        help='the number of epoch between validation')

    parser.add_argument('--ema-decay', default=0.97, type=float,
                        help='hyper-parameter for updating teacher model')
    parser.add_argument('--weight-ramup', default=20, type=int,
                        help='hyper-parameter for ramup')
    parser.add_argument('--mask-ratio', default=0.3, type=float,
                        help='The mask ratio of the Augmentation technique')
    parser.add_argument('--mask-size', default=32, type=int,
                        help='The size of the mask square')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    trainer = Reg_Trainer(args)
    trainer.setup()
    trainer.train()
