import torch
import numpy as np

from torch import nn
from time import time
from prdc import compute_prdc

from utils.data import *
from utils.model import *
from utils.cnn_layers import *
from utils.common_utils import *
from torch.utils.data import DataLoader
    
    
def ham(y_true, y_pred):
    nom = torch.sum(torch.logical_xor(y_true, y_pred))
    denom = y_true.shape.numel()
    
    return nom / denom


def acc(y_true, y_pred):
    nom = torch.sum(y_true == y_pred)
    denom = y_true.shape[0] * y_true.shape[1]
    
    return nom / denom


def loop_score(data, model, center):
    dist_list = []

    model.eval()
    with torch.no_grad():
        for i in range(data.shape[0]):
            inputs = torch.from_numpy(get_xor_corr(data[i]))
            z = model(inputs.float())

            # compute distance
            dist = z - center
            dist = dist.square().mean()
            dist_list.append(dist.item())
                
    return np.mean(dist_list)


def compute_prd(train_data, gen_data, k, repeat):
    metrics = {}
    metrics['precision'] = []
    metrics['recall'] = []
    metrics['density'] = []
    metrics['coverage'] = []
    
    params = {'batch_size': 4096, 'shuffle': False}
    
    train_data = np.transpose(train_data, (0, 2, 1))
    gen_data = np.transpose(gen_data, (0, 2, 1))
    
    start_time = time()
    for i in range(repeat):
        rand_idx = np.random.choice(np.arange(train_data.shape[0]), 10000, replace=False)
        
        train_set = DataLoader(DatasetSampler(train_data[rand_idx]), **params)
        gen_set = DataLoader(DatasetSampler(gen_data[rand_idx]), **params)
    
        train_embed, gen_embed = embed_data(train_set, gen_set)
        results = compute_prdc(real_features=train_embed,
                               fake_features=gen_embed,
                               nearest_k=k)
        
        metrics['precision'].append(results['precision'])
        metrics['recall'].append(results['recall'])
        metrics['density'].append(results['density'])
        metrics['coverage'].append(results['coverage'])
        
        print('%d iter (%0.3f sec)' % (i+1, time()-start_time))
        start_time = time()
        
    return metrics
        
        
def embed_data(train_set, gen_set):
    # init model
    encoder = CNN_Metric(8, 57)
    encoder.eval()
    
    # forward
    train_embed = []
    for batch_idx, x_train in enumerate(train_set):
        z_train = encoder(x_train)
        train_embed.append(z_train.data.cpu().numpy())
    
    gen_embed = []
    for batch_idx, x_gen in enumerate(gen_set):
        z_gen = encoder(x_gen)
        gen_embed.append(z_gen.data.cpu().numpy())
    
    train_embed = np.vstack(train_embed)
    gen_embed = np.vstack(gen_embed)
    
    return train_embed, gen_embed

    
def unique_pitch(pianoroll):
    total_num = pianoroll.shape[0]
    
    count = 0
    num_bar = 8
    num_note = 16
    
    for i in range(total_num):
        count_per_bar = 0
        for j in range(num_bar):
            bar = pianoroll[i][j*num_note:(j+1)*num_note]
            count_per_bar += np.unique(np.where(bar == 1)[1]).shape[0]
            
        count_per_bar = count_per_bar / num_bar
        count += count_per_bar
    
    return count / total_num


def note_density(pianoroll):
    total_num = pianoroll.shape[0]
    
    count = 0
    num_bar = 8
    num_note = 16
    
    for i in range(total_num):
        count_per_bar = 0
        for j in range(num_bar):
            bar = pianoroll[i][j*num_note:(j+1)*num_note]
            count_per_bar += np.sum(np.sum(bar, axis=1) > 0)
            
        count_per_bar = count_per_bar / num_bar
        count += count_per_bar
    
    return count / total_num