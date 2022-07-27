import torch
import numpy as np
from torch import nn
from time import time
    

class LSTM_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTM_Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.logits = nn.Linear(hidden_size, vocab_size)
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        
    def forward(self, x):
        x = self.embed(x)
        x, (h, c) = self.lstm(x)
        logits = self.logits(x.squeeze())
        
        return logits
    
    def predict(self, x, h, c):
        x = self.embed(x)
        x, (h, c) = self.lstm(x, (h, c))
        logits = self.logits(x.squeeze())
        
        return logits, h, c
    
    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    
def sampling_code(batch_size, num_data, AR_model, 
                  num_classes, prob_x1, device, 
                  len_code=32, top_k=0, top_p=0, temp=1):
    
    assert num_data >= batch_size, 'num_data should be larger than batch_size'
    
    num_step = num_data // batch_size
    num_remain = num_data % batch_size
    
    gen_code = np.zeros((num_data, len_code))
    
    AR_model.to(device)
    AR_model.eval()
    
    start_time = time()
    with torch.no_grad():
        for i in range(num_step):
            start_idx = batch_size * i
            end_idx = batch_size * (i + 1)
            
            # init start_token
            start_token = np.random.choice(np.arange(num_classes), batch_size, 
                                           p=prob_x1, replace=True)
            
            inputs = torch.from_numpy(start_token)
            gen_code[start_idx:end_idx, 0] = inputs
            
            inputs = inputs.type(torch.long).unsqueeze(1)
            inputs = inputs.to(device)
            
            # init hidden states
            h, c = AR_model.init_state(batch_size)
            h = h.to(device)
            c = c.to(device)
            
            # AR sampling
            for j in range(1, len_code):
                label, h, c = AR_model(inputs, h, c, temp=temp, top_k=top_k, top_p=top_p)
                gen_code[start_idx:end_idx, j] = np.squeeze(label.data.cpu().numpy())
                inputs = label
            
        # for batch remainder
        start_idx = batch_size * (i + 1)
        start_token = np.random.choice(np.arange(num_classes), num_remain, 
                                       p=prob_x1, replace=True)
        
        inputs = torch.from_numpy(start_token)
        gen_code[start_idx:, 0] = inputs
        
        inputs = inputs.type(torch.long).unsqueeze(1)
        inputs = inputs.to(device)
        
        # init hidden states
        h, c = AR_model.init_state(num_remain)
        h = h.to(device)
        c = c.to(device)
          
        # AR sampling
        for j in range(1, len_code):
            label, h, c = AR_model(inputs, h, c, temp=temp, top_k=top_k, top_p=top_p)
            gen_code[start_idx:, j] = np.squeeze(label.data.cpu().numpy())
            inputs = label
        
    return gen_code


def sampling_from_code(batch_size, gen_code, VQVAE_model, device):
    num_data = gen_code.shape[0]
    num_step = num_data // batch_size
    
    assert num_data >= batch_size, 'num_data should be larger than batch_size'
    gen_data = np.zeros((num_data, 128, 57))
    
    VQVAE_model.to(device)
    VQVAE_model.eval()
    
    start_time = time()
    with torch.no_grad():
        for i in range(num_step):
            inputs = torch.from_numpy(gen_code[batch_size*i:batch_size*(i+1)])
            inputs = inputs.type(torch.long).to(device)
            
            # VQ-VAE sampling
            gen_batch, _ = VQVAE_model.decode_code(inputs)
            gen_batch = gen_batch.data.cpu().numpy().transpose((0, 2, 1))
            gen_data[batch_size*i:batch_size*(i+1)] = gen_batch

        # for batch remainder
        inputs = torch.from_numpy(gen_code[batch_size*(i+1):])
        inputs = inputs.type(torch.long).to(device)
        
        # VQ-VAE sampling
        gen_batch, _ = VQVAE_model.decode_code(inputs)
        gen_batch = gen_batch.data.cpu().numpy().transpose((0, 2, 1))
        gen_data[batch_size*(i+1):] = gen_batch
        
    return gen_data