# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Email   : 
# @File    : 
# @Software: 


import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def load_image(file_path, image_size=512, is_gray=False):
    '''
    读取图像，是否做增强
    '''
    img = Image.open(file_path)
    if is_gray is False and img.mode != 'RGB':
        img = img.convert('RGB')

    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BICUBIC)
    return img


class MyDataset(Dataset):
    '''
    preprocess dataset
    '''
    def __init__(self, opt, mode):
        super(MyDataset, self).__init__()

        self.root_path = opt.dataroot + mode + '/'
        paths = [os.path.join(self.root_path, p) for p in os.listdir(self.root_path)]

        self.paths = []
        self.label = []
        for rp in paths:
            sub_path = os.listdir(rp)
            sub_path.sort(key=lambda x: int(x[:-4]))
            self.paths.extend([os.path.join(rp, s) for s in sub_path])
            self.label.extend([0 if mode == 'train' or rp.split('/')[-1] == 'good' else 1] * len(sub_path))

        self.mode = mode
        self.image_size = opt.img_size

        self.train_transform = transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                                   transforms.RandomHorizontalFlip(0.5),
                                                   transforms.RandomVerticalFlip(0.5),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=opt.data_mean, std=opt.data_std),
                                                   ])
        self.valid_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=opt.data_mean, std=opt.data_std),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.paths[index]
        image = load_image(path, self.image_size)
        if self.mode == 'train':
            image = self.train_transform(image)
            mask = 0
        else:
            image = self.valid_transform(image)
            mask_path = path.replace('test', 'ground_truth').replace('.png', '_mask.png')
            if os.path.exists(mask_path):
                mask = load_image(mask_path, self.image_size, is_gray=True)
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros((1, self.image_size, self.image_size))

        return image, mask, self.label[index]

    def __len__(self):
        return len(self.paths)


class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


def load_data(opt):
    train_ds = MyDataset(opt, 'train')
    valid_ds = MyDataset(opt, 'test')
    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batch_size, shuffle=True, drop_last=False, num_workers=4)

    return Data(train_dl, valid_dl)

