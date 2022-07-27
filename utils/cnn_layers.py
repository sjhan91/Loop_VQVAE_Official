import torch
import numpy as np

from torch import nn
from utils.dl_utils import *


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(CNNBlock, self).__init__()
        
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2))
                  
    def forward(self, x):
        return self.cnn_block(x)

    
class ResBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, padding=1):
        super(ResBlock, self).__init__()
        
        self.res_block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(ch),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(ch))
                  
    def forward(self, x):
        out = self.res_block(x)
        out = out + x
        
        return out
    
    
class CNN_Encoder(nn.Module):
    def __init__(self, ch, num_pitch, latent_dim):
        super(CNN_Encoder, self).__init__()
        
        self.first_layer = CNNBlock(num_pitch, ch, kernel_size=3, padding=1)
        self.cnn_layer = nn.ModuleList([CNNBlock(ch, ch, kernel_size=3, stride=2, padding=1) for i in range(2)])
        self.res_layer = nn.ModuleList([ResBlock(ch, kernel_size=3, padding=1) for i in range(2)])
            
        self.mu = nn.Linear(32*ch, latent_dim)
        self.std = nn.Linear(32*ch, latent_dim)
        
    def encode(self, x):
        x = self.first_layer(x)
        
        for i in range(2):
            x = self.cnn_layer[i](x)
            x = self.res_layer[i](x)
            
        x = x.view(x.shape[0], -1)
        
        mu = self.mu(x)
        std = nn.Softplus()(self.std(x))
        
        # reparam
        z = self.reparameterize(mu, std)
        
        return z, mu, std
    
    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)

        return mu + (eps * std)
    
    def forward(self, x):
        z, mu, std = self.encode(x)
        
        return z, mu, std

    
class CNN_Decoder(nn.Module):
    def __init__(self, ch, num_pitch, latent_dim):
        super(CNN_Decoder, self).__init__()
        
        self.ch = ch
        self.latent_dim = latent_dim
        
        self.first_layer = nn.Linear(latent_dim, 32*ch)
        self.cnn_layer = nn.ModuleList([CNNBlock(ch, ch, kernel_size=3, stride=1, padding=1) for i in range(2)])
        self.res_layer = nn.ModuleList([ResBlock(ch, kernel_size=3, padding=1) for i in range(2)])
        self.final_layer = nn.Conv1d(ch, num_pitch, 7, stride=1, padding=3)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.first_layer(x)
        x = nn.LeakyReLU(0.2)(x)
        
        x = x.reshape(batch_size, self.ch, x.shape[1]//self.ch)
        
        for i in range(2):
            x = nn.Upsample(scale_factor=2)(x)
            x = self.cnn_layer[i](x)
            x = self.res_layer[i](x)
        
        x = self.final_layer(x)
        
        logits = nn.Sigmoid()(x)
        label = (logits >= 0.5).to(torch.float32)
                
        return label, logits
    
    
class CNN_Metric(nn.Module):
    def __init__(self, ch, num_pitch):
        super(CNN_Metric, self).__init__()
        
        self.cnn_layer_1 = CNNBlock(num_pitch, ch, kernel_size=3, stride=2, padding=1)
        self.cnn_layer_2 = CNNBlock(ch, ch, kernel_size=3, stride=4, padding=1)
        self.linear = nn.Linear(16*ch, 16*ch)
    
    def forward(self, x):
        x = self.cnn_layer_1(x)
        x = self.cnn_layer_2(x)
        
        x = x.view(x.shape[0], -1)
        z = self.linear(x)
        
        return z