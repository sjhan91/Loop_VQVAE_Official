import torch
import torch.optim as optim
import pytorch_lightning as pl

from torch import nn
from utils.loss import *
from utils.metric import *
from utils.dl_utils import *

from utils.cnn_layers import *
from utils.lstm_layers import *
from utils.vqvae_layers import *


class AutoEncoder(pl.LightningModule):
    def __init__(self, num_features):
        super().__init__()
        self.automatic_optimization = False
        
        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_features//2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(num_features//2, num_features//2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(num_features//2, num_features//4, bias=False))
        
        self.decoder = nn.Sequential(
            nn.Linear(num_features//4, num_features//2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(num_features//2, num_features//2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(num_features//2, num_features, bias=False))

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1e3, eta_min=5e-6)
        
        return [opt], [sch]

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        
        x = train_batch
        z = self.encoder(x)
        x_recon = self.decoder(z)
        loss = nn.MSELoss()(x_recon, x)
        self.log('train_loss', loss, prog_bar=True)
        
        self.manual_backward(loss)
        opt.step()
        
        if self.trainer.is_last_batch:
            sch.step()
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        z = self.encoder(x)
        x_recon = self.decoder(z)
        loss = nn.MSELoss()(x_recon, x)
        self.log('val_loss', loss, prog_bar=True)

        return loss


class SVDD(pl.LightningModule):
    def __init__(self, encoder, center):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = encoder
        self.center = center

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1e3, eta_min=5e-6)
        
        return [opt], [sch]

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        
        x = train_batch
        batch_size = x.shape[0]
        
        z = self.encoder(x)
        loss = nn.MSELoss()(z, self.center.repeat(batch_size, 1))
        self.log('train_loss', loss, prog_bar=True)
        
        self.manual_backward(loss)
        opt.step()
        
        if self.trainer.is_last_batch:
            sch.step()
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        batch_size = x.shape[0]
        
        z = self.encoder(x)
        loss = nn.MSELoss()(z, self.center.repeat(batch_size, 1))
        self.log('val_loss', loss, prog_bar=True)

        return loss
    
    
class CNN_VAE(pl.LightningModule):
    def __init__(self, ch, num_pitch, latent_dim):
        super().__init__()
        self.automatic_optimization = False
        
        self.encoder = CNN_Encoder(ch, num_pitch, latent_dim)
        self.decoder = CNN_Decoder(ch, num_pitch, latent_dim)
        self.beta = 0

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        label, logits = self.decoder(z)
        
        return label, logits

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=5e-6)
        
        return [opt], [sch]

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        
        x = train_batch
        # x = data_aug(x)
        
        z, mu, sigma = self.encoder(x)
        label, logits = self.decoder(z)
        
        # loss
        self.beta = kl_annealing(self.current_epoch, 0, 1)
        loss = vae_loss(logits, x, mu, sigma, self.beta)
        hamming = ham(x, label).item()
        
        self.log('train_loss', hamming, prog_bar=True)
        
        self.manual_backward(loss)
        opt.step()
        
        if self.trainer.is_last_batch:
            sch.step()
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        z, mu, sigma = self.encoder(x)
        label, logits = self.decoder(z)
        
        loss = vae_loss(logits, x, mu, sigma, self.beta)
        hamming = ham(x, label).item()
        
        self.log('val_loss', hamming, prog_bar=True)

        return loss


class VQVAE(pl.LightningModule):
    def __init__(self, ch, num_pitch, latent_dim, num_embed, thres=1):
        super().__init__()
        self.automatic_optimization = False
        
        self.encoder = VQVAE_Encoder(ch, num_pitch, latent_dim)
        self.quantize = Quantize(latent_dim, num_embed, thres)
        self.decoder = VQVAE_Decoder(ch, num_pitch, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        quant_z, quant_idx, perplex = self.quantize(x)
        label, logits = self.decoder(quant_z)

        return label, logits

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=5e-6)
        
        return [opt], [sch]
    
    def decode_code(self, quant_idx):
        z = self.quantize.embed_code(quant_idx)
        z = z.transpose(1, 2)

        return self.decoder(z)

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        
        x = train_batch
        # x = data_aug(x)
        
        z = self.encoder(x)
        quant_z, _, perplex = self.quantize(z)
        label, logits = self.decoder(quant_z)
        
        # loss
        recon_loss_ = recon_loss(logits, x)
        commit_loss_ = commit_loss(z, quant_z.detach())
        
        loss = recon_loss_ + (0.25 * commit_loss_)
        hamming = ham(x, label).item()
        
        self.log('train_loss', hamming, prog_bar=True)
        
        self.manual_backward(loss)
        opt.step()
        
        if self.trainer.is_last_batch:
            sch.step()
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        z = self.encoder(x)
        quant_z, _, perplex = self.quantize(z)
        label, logits = self.decoder(quant_z)
        
        # loss
        recon_loss_ = recon_loss(logits, x)
        commit_loss_ = commit_loss(z, quant_z.detach())
        
        loss = recon_loss_ + (0.25 * commit_loss_)
        hamming = ham(x, label).item()
        
        self.log('val_loss', hamming, prog_bar=True, sync_dist=True)

        return loss
    
    
class LSTM(pl.LightningModule):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.automatic_optimization = False
        
        self.inf = -float('Inf')
        self.vocab_size = vocab_size
        self.decoder = LSTM_Decoder(embed_size, hidden_size, vocab_size, num_layers=num_layers)

    def forward(self, x, h, c, temp=1, top_k=0, top_p=0):
        logits, h, c = self.decoder.predict(x, h, c)
        
        if top_k > 0 and top_p > 0:
            raise AssertionError('wrong sampling strategy!')
        
        # top-k sampling
        if top_k > 0:
            remove_idx = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits[remove_idx] = self.inf
        
        # top-p sampling
        if top_p > 0:
            sorted_logits, sorted_index = torch.sort(logits, descending=True)
            cum_prob = torch.cumsum(nn.Softmax(dim=1)(logits), dim=-1) 
            to_remove = (cum_prob > top_p)
            
            # shift right
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = 0
            
            remove_idx = to_remove.scatter(dim=1, index=sorted_index, src=to_remove)
            logits[remove_idx] = self.inf
         
        # sampling from distribution
        prob = nn.Softmax(dim=1)(logits / temp)
        label = torch.multinomial(prob, 1)
        
        return label, h, c

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=5e-6)
        
        return [opt], [sch]
    
    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        
        inputs = train_batch[:, :-1]
        outputs = train_batch[:, 1:]

        logits = self.decoder(inputs)
        logits_2d = torch.reshape(logits, (-1, self.vocab_size))
        outputs_1d = torch.reshape(outputs, (-1,))
          
        # loss
        loss = ce_loss(logits_2d, outputs_1d)
        
        prob = nn.Softmax(dim=2)(logits)
        label = torch.argmax(prob, 2)
        
        accuracy = acc(outputs, label).item()
        self.log('train_loss', accuracy, prog_bar=True)
        
        self.manual_backward(loss)
        opt.step()
        
        if self.trainer.is_last_batch:
            sch.step()
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        inputs = val_batch[:, :-1]
        outputs = val_batch[:, 1:]
        
        logits = self.decoder(inputs)
        logits_2d = torch.reshape(logits, (-1, self.vocab_size))
        outputs_1d = torch.reshape(outputs, (-1,))
          
        # loss
        loss = ce_loss(logits_2d, outputs_1d)
        
        prob = nn.Softmax(dim=2)(logits)
        label = torch.argmax(prob, 2)
        
        accuracy = acc(outputs, label).item()
        self.log('val_loss', accuracy, prog_bar=True, sync_dist=True)

        return loss
    
    def init_state(self, batch_size):
        return self.decoder.init_state(batch_size)