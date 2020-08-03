import os,sys,signal
import math

import pickle
import numpy as np                                       # fast vectors and matrices
import matplotlib.pyplot as plt                          # plotting
import argparse

import torch
from torch.nn.functional import conv1d, mse_loss
import torch.nn.functional as F
import torch.nn as nn
from nnAudio import Spectrogram
from time import time

# For the dataloading
from Dataset import MusicNet_nnAudio
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument("factor", type=int,help="factor for STFT:")
parser.add_argument('-d', '--device', type=int, help="choose device")
parser.add_argument('-n', '--num_workers', type=int, help="choose device")
args = parser.parse_args()

epoches = 100
num_workers = args.num_workers

device = f'cuda:{args.device}'
batch_size = 16

start = time()
train_set = MusicNet_nnAudio(path='./data/', groups=['train'], sequence_length=327680, refresh=True)
loading_time = time()-start

train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, num_workers=num_workers)

factor = args.factor
n_fft = 4096//factor
lr = 1e-4




Loss = torch.nn.BCELoss()
def L(yhatvar,y):
    return Loss(yhatvar,y)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        f_kernal = 128//factor
        self.STFT_layer = Spectrogram.STFT(sr=44100, n_fft=n_fft, hop_length=HOP_LENGTH, center=True, device=device)
        self.freq_cnn1 = torch.nn.Conv2d(1,4, (f_kernal,3), stride=(8,1), padding=1)
        self.freq_cnn2 = torch.nn.Conv2d(4,8, (f_kernal,3), stride=(8,1), padding=1)
        shape = self.shape_inference(f_kernal)
        self.bilstm = torch.nn.LSTM(shape*8, shape*8, batch_first=True, bidirectional=True)
        self.pitch_classifier = torch.nn.Linear(shape*8*2, 88)

    def shape_inference(self, f_kernal):
        layer1 = (n_fft//2+2-(f_kernal))//8 + 1 
        layer2 = (layer1+2-(f_kernal))//8 + 1 
        return layer2
    
    def forward(self,x):
        x = self.STFT_layer(x[:,:-1])
        x = torch.log(x+1e-5)
        x = torch.relu(self.freq_cnn1(x.unsqueeze(1)))
        x = torch.relu(self.freq_cnn2(x))
        x, _ = self.bilstm(x.view(x.size(0), x.size(1)*x.size(2), x.size(3)).transpose(1,2))
        x = torch.sigmoid(self.pitch_classifier(x))
        
        return x
    
model = Model()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn  = torch.nn.BCELoss()


times = []
loss_histroy = []
print("epoch\ttrain loss\ttime")
total_i = len(train_loader)
for e in range(epoches):
    running_loss = 0
    start = time()
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x = data['audio'].to(device)
        y = data['frame'].to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()        
        
        print(f"Training {idx+1}/{total_i} batches\t Loss: {loss.item()}", end = '\r')
    time_used = time()-start
    times.append(time_used)
    print(' '*200, end='\r')
    print(f'{e+1}\t{running_loss/total_i:.6f}\t{time_used:.6f}')
    loss_histroy.append(running_loss/total_i)
    
nnAudio_result = {}
nnAudio_result['loss_histroy'] = loss_histroy
nnAudio_result['time_histroy'] = times
nnAudio_result['loading_time'] = loading_time

import pickle
with open(f'./result/nnAudio_result_STFT-{n_fft}', 'wb') as f:
    pickle.dump(nnAudio_result,f)
    
torch.save(model.state_dict(), f'./weight/nnAudio_result_{n_fft}')
