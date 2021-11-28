# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Email   : 
# @File    : 
# @Software: 


import torch
from torch import nn

from memory_module import Encoder, Decoder, COMP_EMA, weights_init


class MemoryModule(torch.nn.Module):
    def __init__(self, input_dim=3, dim=512, K=100, commitment_cost=0.25):
        '''
        :param input_dim:
        :param dim: size of the latent vectors
        :param K: number of latent vectors
        '''
        super(MemoryModule, self).__init__()

        num_residual_hiddens = 32
        num_residual_layers = 2
        self.encoder = Encoder(input_dim, dim,
                               num_residual_layers,
                               num_residual_hiddens)

        self.codebook = COMP_EMA(K, dim, commitment_cost)

        self.decoder = Decoder(dim,
                               dim,
                               num_residual_layers,
                               num_residual_hiddens)

        self.apply(weights_init)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.codebook(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity

