import itertools

import cv2
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import h5py
import math

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class Crowd(data.Dataset):
    def __init__(self, root, crop_size, downsample_ratio, method='train', info=None):
        self.im_list = sorted(glob(os.path.join(root, 'images/*.jpg')))
        if method not in ['train', 'val']:
            raise Exception('Method is not implemented!')
        self.label_list = []
        if method == 'train':
            try:
                with open(info) as f:
                    for i in f:
                        self.label_list.append(i.strip())
            except:
                raise Exception("please give right info")

            labeled = []
            for i in self.im_list:
                if os.path.basename(i) in self.label_list:
                    labeled.append(1)
                else:
                    labeled.append(0)
            labeled = np.array(labeled)
            self.labeled_idx = np.where(labeled == 1)[0]
            self.unlabeled_idx = np.where(labeled == 0)[0]

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        self.root = root
        self.method = method
        assert self.c_size % self.d_ratio == 0
        self.w_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.s_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):

        im_path = self.im_list[item]
        name = os.path.basename(im_path).split('.')[0]
        gd_path = os.path.join(self.root, 'gt_points', '{}.npy'.format(name))
        img = Image.open(im_path).convert('RGB')
        keypoints = np.load(gd_path)

        if self.method == 'train':
            den_map_path = os.path.join(self.root, 'gt_den', '{}.h5'.format(name))
            den_map = h5py.File(den_map_path, 'r')['density_map']
            label = (os.path.basename(im_path) in self.label_list)
            return self.train_transform_density_map(img, den_map, label)

        elif self.method == 'val':
            w, h = img.size
            new_w = math.ceil(w / 32) * 32
            new_h = math.ceil(h / 32) * 32
            img = img.resize((new_w, new_h), Image.BICUBIC)
            return self.w_transform(img), len(keypoints), name

    def train_transform_density_map(self, img, den_map, label):
        wd, ht = img.size
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd * re_size)
        htt = (int)(ht * re_size)
        if min(wdd, htt) >= self.c_size:
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            den_map = cv2.resize(den_map[:, :], (wd, ht), interpolation=cv2.INTER_CUBIC) / (re_size ** 2)

        st_size = min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        den_map = den_map[i: (i + h), j: (j + w)]
        den_map = den_map.reshape([h // self.d_ratio, self.d_ratio, w // self.d_ratio, self.d_ratio]).sum(axis=(1, 3))

        if random.random() > 0.5:
            img = F.hflip(img)
            den_map = np.fliplr(den_map)

        return self.w_transform(img), self.s_transform(img), torch.from_numpy(den_map.copy()).float().unsqueeze(0), label


class TwoStreamBatchSampler(data.Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


