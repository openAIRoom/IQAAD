# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Email   : 
# @File    : 
# @Software: 


import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def unorm_image(image, mean, std):
    with torch.no_grad():
        img = image.clone()
        for i, (m, s) in enumerate(zip(mean, std)):
            img[:, i, :, :].mul_(s).add_(m)
        return img


def to_gray(tensor):
    '''
    3通道的RGB图像转为1通道的灰度图像
    '''
    R, G, B = tensor[:, 0, ...], tensor[:, 1, ...], tensor[:, 2, ...]
    # gray = 0.299 * R + 0.587 * G + 0.114 * B
    gray = (R + G + B) / 3
    gray = gray.unsqueeze(1)
    return gray


def replicate_gray(tensor):
    '''
    变为灰度图像并扩展为3通道
    '''
    return to_gray(tensor).repeat(1, 3, 1, 1)


def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


############################# 显示 ########################################
def plot_current_errors(epoch, errors, viz, plot_errors):
    plot_errors['X'].append(epoch)
    plot_errors['Y'].append([np.mean(errors[k]) for k in plot_errors['legend']])
    viz.line(
        X=np.stack([np.array(plot_errors['X'])] * len(plot_errors['legend']), 1),
        Y=np.array(plot_errors['Y']),
        opts={
            'title': 'train loss over time',
            'legend': plot_errors['legend'],
            'xlabel': 'Epoch',
            'ylabel': 'Loss',
            'ylim': [0, 100]
        },
        win=0
    )


def plot_current_roc(epoch, rocauc, viz, plot_rocauc, k=1):
    plot_rocauc['X'].append(epoch)
    plot_rocauc['Y'].append([np.mean(rocauc[k]) for k in plot_rocauc['legend']][0])
    viz.line(
        X=np.array(plot_rocauc['X']),
        Y=np.array(plot_rocauc['Y']),
        opts={
            'title': 'test rocauc_image over time' if k == 1 else 'test rocauc_pixel over time',
            'legend': plot_rocauc['legend'],
            'xlabel': 'Epoch',
            'ylabel': 'rocauc',
            'ylim': [0, 1.2]
        },
        win=k
    )


#############################################################################
def load_model(memory_model, pretrained):
    assert "model/" + pretrained, '未找到模型'
    weights = torch.load("model/" + pretrained)

    pretrained_memory_model_dict = weights['memory_model']
    memory_model_dict = memory_model.state_dict()
    pretrained_memory_model_dict = {k: v for k, v in pretrained_memory_model_dict.items() if k in memory_model_dict}
    memory_model_dict.update(pretrained_memory_model_dict)
    memory_model.load_state_dict(memory_model_dict)


def save_model(memory_model, epoch):
    state = {'memory_model': memory_model.state_dict()}

    model_out_path = "model/" + "model_epoch_{0}.pth".format(epoch)
    if not os.path.exists("model/"):
        os.makedirs("model/")

    # torch.save(state, model_out_path)
    torch.save(state, model_out_path, _use_new_zipfile_serialization=False)  #保存的pytorch1.6模型转给1.4用
    print("Model saved to {0}".format(model_out_path))