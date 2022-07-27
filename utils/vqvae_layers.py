import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from .cnn_layers import *
from utils.dl_utils import *


class VQVAE_Encoder(nn.Module):
    def __init__(self, ch, num_pitch, latent_dim):
        super(VQVAE_Encoder, self).__init__()
        
        self.first_layer = CNNBlock(num_pitch, ch, kernel_size=3, padding=1)
        self.cnn_layer = nn.ModuleList([CNNBlock(ch, ch, kernel_size=3, stride=2, padding=1) for i in range(2)])
        self.res_layer = nn.ModuleList([ResBlock(ch, kernel_size=3, padding=1) for i in range(2)])
        self.last_layer = nn.Conv1d(ch, latent_dim, kernel_size=3, padding=1)
        
    def encode(self, x):
        x = self.first_layer(x)
        
        for i in range(2):
            x = self.cnn_layer[i](x)
            x = self.res_layer[i](x)
            
        x = self.last_layer(x)
        
        return x
    
    def forward(self, x):
        x = self.encode(x)
        
        return x


class VQVAE_Decoder(nn.Module):
    def __init__(self, ch, num_pitch, latent_dim):
        super(VQVAE_Decoder, self).__init__()
        
        self.first_layer = CNNBlock(latent_dim, ch, kernel_size=3, padding=1)
        self.cnn_layer = nn.ModuleList([CNNBlock(ch, ch, kernel_size=3, stride=1, padding=1) for i in range(2)])
        self.res_layer = nn.ModuleList([ResBlock(ch, kernel_size=3, padding=1) for i in range(2)])
        self.final_layer = nn.Conv1d(ch, num_pitch, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.first_layer(x)
        x = nn.LeakyReLU(0.2)(x)
        
        for i in range(2):
            x = nn.Upsample(scale_factor=2)(x)
            x = self.cnn_layer[i](x)
            x = self.res_layer[i](x)
        
        x = self.final_layer(x)
        
        logits = nn.Sigmoid()(x)
        label = (logits >= 0.5).to(torch.float32)
                
        return label, logits


class Quantize(nn.Module):
    def __init__(self, latent_dim, num_embed, thres=1, decay=0.99, eps=1e-5):
        super(Quantize, self).__init__()
        
        self.eps = eps
        self.thres = thres
        self.decay = decay

        self.num_embed = num_embed
        self.latent_dim = latent_dim

        self.register_buffer('embed', torch.randn(latent_dim, num_embed))
        self.register_buffer('embed_sum', torch.zeros(latent_dim, num_embed))
        self.register_buffer('cluster_size', torch.zeros(num_embed))
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x_flat = x.detach().reshape(-1, self.latent_dim)
        
        # x^2 -2xy + y^2
        dist = (x_flat.pow(2).sum(1, keepdim=True)
                - 2 * x_flat @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True))
        
        embed_idx = torch.argmin(dist, 1)
        embed_onehot = F.one_hot(embed_idx, self.num_embed).type(x.dtype)
        
        # get quantized embedding from dict
        quantize = self.embed_code(embed_idx)
        quantize = quantize.view_as(x)
        embed_idx = embed_idx.reshape(x.shape[:2])
        
        # dict update
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = x_flat.transpose(0, 1) @ embed_onehot
            
            self.cluster_size = self.ema(self.cluster_size, embed_onehot_sum)
            
            # cluster_size correction
            n = self.cluster_size.sum()
            self.cluster_size = (self.cluster_size + self.eps) / (n + self.num_embed * self.eps) * n
            
            self.embed_sum = self.ema(self.embed_sum, embed_sum)
            self.embed = self.embed_sum / self.cluster_size.unsqueeze(0)
            
            # random restart
            usage = (self.cluster_size.view(1, self.num_embed) >= self.thres).float()
            
            # update codebook with random restart
            rand_idx = np.random.randint(x_flat.shape[0], size=self.num_embed)
            self.embed = usage * self.embed + (1 - usage) * x_flat.T[:, rand_idx]
            
        # straight-through estimator
        quantize = x + (quantize - x).detach()
        
        # perplexity
        probs = embed_onehot.mean(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        perplexity = torch.exp(entropy)
        
        return quantize.transpose(1, 2), embed_idx, perplexity
    
    def ema(self, old, new):
        return (self.decay * old) + ((1 - self.decay) * new)
    
    def embed_code(self, embed_idx):
        return F.embedding(embed_idx, self.embed.transpose(0, 1))