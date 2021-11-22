# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Email   : 
# @File    : 
# @Software: 


import os
import argparse
import numpy as np
from tqdm import tqdm
import visdom
from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

from model import MemoryModule
from dataloader import load_data
from utils import load_model, save_model, plot_current_roc, plot_current_errors, \
    unorm_image, mean_smoothing, to_gray, replicate_gray

from anomaly_score.ms_fsim_score import MSFSIM
from anomaly_score.ms_gmsd_score import MSGMSD
from anomaly_score.ms_ssim_our import MSSSIM

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_arg():
    parser = argparse.ArgumentParser('Parameters')
    parser.add_argument('--dataroot', default="G:/Datasets/MVTec/mvtec_anomaly_detection/zipper/", type=str, help='data path')
    parser.add_argument('--img_size', default=256, type=int, help='image size')
    parser.add_argument('--nc', default=3, type=int, help='channel of input image')
    parser.add_argument('--dim', type=int, default=512, help='size of the latent vectors')
    parser.add_argument('--K', type=int, default=100, help='number of latent vectors')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size')

    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    parser.add_argument('--n_epoch', type=int, default=2000, help='epoch')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start from epoch i')
    parser.add_argument('--data_mean', type=list, default=[0.485, 0.456, 0.406], help='')
    parser.add_argument('--data_std', type=list, default=[0.229, 0.224, 0.225], help='')

    parser.add_argument('--loss_beta', type=float, default=1.0, help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--loss_gms', type=float, default=1, help='weight of the gms loss')
    parser.add_argument('--loss_ssim', type=float, default=1, help='weight of the ssim loss')

    parser.add_argument("--pretrained", default="", type=str, help="load model")
    parser.add_argument("--outfile", default="results/", type=str, help="outfile")

    return parser.parse_known_args()[0]


def train(epoch, memory_model, train_loader, errors):
    memory_model.train()

    for iteration, (datas, masks, labels) in enumerate(tqdm(train_loader)):
        datas = datas.to(device)

        optimizer.zero_grad()
        loss_vq, recon, perplexity = memory_model(datas)

        # loss
        l2_loss = mse(datas, recon)
        msssim_loss = msssim(datas, recon, as_loss=True)
        msgmsd_loss = msgmsd(datas, recon, as_loss=True)
        msfsim_loss = ms_fsim_model(replicate_gray(datas), replicate_gray(recon), as_loss=True)

        loss = l2_loss + \
               loss_vq + \
               msgmsd_loss + msssim_loss + msfsim_loss
        loss.backward()
        optimizer.step()

        errors['train_loss_l2'].append(l2_loss.item())
        errors['train_loss_vq'].append(loss_vq.item())
        errors['train_loss_msssim'].append(msssim_loss.item())
        errors['train_loss_msgmsd'].append(msgmsd_loss.item())
        errors['train_loss_msfsim'].append(msfsim_loss.item())

    if epoch % 10 == 0:
        msfsim_anomaly_map = ms_fsim_model(datas, recon, as_loss=False)
        msfsim_anomaly_map_25 = mean_smoothing(msfsim_anomaly_map, kernel_size=25)

        save_maps = []
        [save_maps.extend([unorm_image(datas, opt.data_mean, opt.data_std)[i:i+1],
                           unorm_image(recon, opt.data_mean, opt.data_std)[i:i+1],
                           msfsim_anomaly_map_25[i:i+1].repeat(1, 3, 1, 1)]) for i in range(min(3, datas.shape[0]))]
        save_image(torch.cat(save_maps, dim=0).data.cpu(),
                   '{}/image_{}_m.jpg'.format(opt.outfile + 'train', epoch), nrow=3)


def test(epoch, memory_model, test_loader, errors, rocauc_image, rocauc_pixel):
    try:
        os.makedirs(opt.outfile + 'test/epoch_' + str(epoch))
    except:
        pass

    memory_model.eval()

    msfsim_scores_maps = []

    gt_list = []
    gt_mask_list = []
    for iteration, (datas, masks, labels) in enumerate(tqdm(test_loader)):
        datas = datas.to(device)

        # to calculate roc
        gt_list.extend(labels.cpu().numpy())
        gt_mask_list.extend(masks.int().cpu().numpy())

        with torch.no_grad():
            loss_vq, recon, perplexity = memory_model(datas)

            msfsim_anomaly_map = ms_fsim_model(datas, recon, as_loss=False)

        msfsim_anomaly_map = mean_smoothing(msfsim_anomaly_map, kernel_size=21)
        msfsim_scores_maps.extend(msfsim_anomaly_map.squeeze(1).detach().cpu().numpy())

        # loss
        l2_loss = mse(datas, recon)
        msssim_loss = msssim(datas, recon, as_loss=True)
        msgmsd_loss = msgmsd(datas, recon, as_loss=True)
        msfsim_loss = ms_fsim_model(replicate_gray(datas), replicate_gray(recon), as_loss=True)

        errors['test_loss_l2'].append(l2_loss.item())
        errors['test_loss_vq'].append(loss_vq.item())
        errors['test_loss_msssim'].append(msssim_loss.item())
        errors['test_loss_msgmsd'].append(msgmsd_loss.item())
        errors['test_loss_msfsim'].append(msfsim_loss.item())

        if iteration < 10:
            save_maps = []
            [save_maps.extend([unorm_image(datas, opt.data_mean, opt.data_std)[i:i + 1],
                               unorm_image(recon, opt.data_mean, opt.data_std)[i:i + 1],
                               msfsim_anomaly_map[i:i + 1].repeat(1, 3, 1, 1)]) for i in
             range(min(3, datas.shape[0]))]
            save_image(torch.cat(save_maps, dim=0).data.cpu(),
                       '{}/image_{}.jpg'.format(opt.outfile + 'test/epoch_' + str(epoch), iteration + 1), nrow=3)

    msfsim_img_roc_auc, msfsim_per_pixel_rocauc = auc_calculate(msfsim_scores_maps, gt_list, gt_mask_list)
    rocauc_image['msfsim_img_rocauc'].append(msfsim_img_roc_auc)
    rocauc_pixel['msfsim_per_pixel_rocauc'].append(msfsim_per_pixel_rocauc)


def auc_calculate(scores, gt_list, gt_mask_list):
    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    # print('image ROCAUC: %.3f' % (img_roc_auc))
    # with open("rocuac.txt", "a") as f:
    #     f.write('%.3f' % img_roc_auc + '\t')

    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    # print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))
    # with open("rocuac.txt", "a") as f:
    #     f.write('%.3f' % per_pixel_rocauc + '\n')

    return img_roc_auc, per_pixel_rocauc


if __name__ == '__main__':
    opt = parse_arg()
    disp_str = ''
    for attr in sorted(dir(opt), key=lambda x: len(x)):
        if not attr.startswith('_'):
            disp_str += ' {} : {}\n'.format(attr, getattr(opt, attr))
    print(disp_str)

    try:
        os.makedirs(opt.outfile + 'train/')
        os.makedirs(opt.outfile + 'test/')
        print('mkdir:', opt.outfile)
    except OSError:
        pass

    seed = np.random.randint(0, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    # --------------build models -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    memory_model = MemoryModule(opt.nc, opt.dim, opt.K).to(device)
    if opt.pretrained:
        load_model(memory_model, opt.pretrained)
        print('Model is loaded!')

    params_memory = list(memory_model.parameters())
    params = params_memory
    optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epoch)

    # -----------------load dataset--------------------------
    data_loader = load_data(opt)

    # Loss Functions
    msssim = MSSSIM()
    msgmsd = MSGMSD(device=device)
    mse = nn.MSELoss(reduction='mean')
    ms_fsim_model = MSFSIM(device=device)

    for epoch in range(opt.start_epoch, opt.n_epoch):
        errors = defaultdict(list)
        rocauc_image = defaultdict(list)
        rocauc_pixel = defaultdict(list)

        train(epoch, memory_model, data_loader.train, errors)
        if epoch % 10 == 0:
            # save_model(memory_model, 10000)
            test(epoch, memory_model, data_loader.valid, errors, rocauc_image, rocauc_pixel)

        scheduler.step()
        # visdom
        if epoch == opt.start_epoch:
            viz = visdom.Visdom(env='main', use_incoming_socket=False)
            plot_errors = {'X': [], 'Y': [], 'legend': list(errors.keys())}
            plot_rocauc_image = {'X': [], 'Y': [], 'legend': list(rocauc_image.keys())}
            plot_rocauc_pixel = {'X': [], 'Y': [], 'legend': list(rocauc_pixel.keys())}
        plot_current_errors(epoch, errors, viz, plot_errors)
        if epoch % 10 == 0:
            plot_current_roc(epoch, rocauc_image, viz, plot_rocauc_image, k=1)
            plot_current_roc(epoch, rocauc_pixel, viz, plot_rocauc_pixel, k=2)
        print('----------------------------------------')
        print('Epoch:', epoch)
        print('train_loss_l2 {:.6f}/ '
              'train_loss_vq {:.6f}/ '
              'train_loss_msssim {:.6f}/ '
              'train_loss_msgmsd {:.6f}/ '
              'train_loss_msfsim {:.6f}'.format(np.mean(errors['train_loss_l2']),
                                               np.mean(errors['train_loss_vq']),
                                               np.mean(errors['train_loss_msssim']),
                                               np.mean(errors['train_loss_msgmsd']),
                                               np.mean(errors['train_loss_msfsim'])), end=' ')
        if epoch % 10 == 0:
            print('test_loss_l2 {:.6f}/ '
                  'test_loss_vq {:.6f}/ '
                  'test_loss_msssim {:.6f}/ '
                  'test_loss_msgmsd {:.6f}/ '
                  'test_loss_msfsim {:.6f}'.format(np.mean(errors['test_loss_l2']),
                                                   np.mean(errors['test_loss_vq']),
                                                   np.mean(errors['test_loss_msssim']),
                                                   np.mean(errors['test_loss_msgmsd']),
                                                   np.mean(errors['test_loss_msfsim'])), end=' ')
            print('\nimage_rocauc {:.6f}/ '
                  'pixel_rocauc {:.6f}/ '.format(np.mean(rocauc_image['msfsim_img_rocauc']),
                                                 np.mean(rocauc_pixel['msfsim_per_pixel_rocauc'])))
        else:
            print('\n')

    print('Training is finished')
