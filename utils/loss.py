import torch
from torch import nn
import numpy as np


def vae_loss(recon_x, x, mu, std, beta=0):
    norm = np.prod(x.shape)
    logvar = std.pow(2).log()
    
    recon = nn.BCELoss(reduction='sum')(recon_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    loss = recon + (beta * kl)
    
    return loss / norm


def bce_loss(y_pred, y_true):
    norm = np.prod(y_true.shape)
    loss = nn.BCELoss(reduction='sum')(y_pred, y_true)
    
    return loss / norm


def ce_loss(y_pred, y_true):
    loss = nn.CrossEntropyLoss()(y_pred, y_true)
    
    return loss


def recon_loss(recon_x, x):
    norm = np.prod(x.shape[1:])
    loss = nn.BCELoss(reduction='sum')(recon_x, x)
    
    return loss / norm


def commit_loss(x1, x2):
    norm = np.prod(x1.shape[1:])
    loss = nn.MSELoss(reduction='sum')(x1, x2)
    
    return loss / norm