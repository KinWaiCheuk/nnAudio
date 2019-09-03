import torch
import torch.nn as nn
from torch.nn.functional import conv1d

import numpy as np
import torch
from time import time
import math
from scipy.signal import get_window
from scipy import fft


sz_float = 4    # size of a float
epsilon = 10e-8 # fudge factor for normalization

## Basic tools for computation ##
def nextpow2(A):
    return int(np.ceil(np.log2(A)))

def complex_mul(cqt_filter, stft):
    cqt_filter_real = cqt_filter[0]
    cqt_filter_imag = cqt_filter[1]
    fourier_real = stft[0]
    fourier_imag = stft[1]
    
    CQT_real = torch.matmul(cqt_filter_real, fourier_real) - torch.matmul(cqt_filter_imag, fourier_imag)
    CQT_imag = torch.matmul(cqt_filter_real, fourier_imag) + torch.matmul(cqt_filter_imag, fourier_real)   
    
    return CQT_real, CQT_imag


## Kernal generation functions ##
def create_fourier_kernals(n_fft, freq_bins=None, low=50,high=6000, sr=44100, freq_scale='linear', windowing="no"):
    if freq_bins==None:
        freq_bins = n_fft//2+1

    s = np.arange(0, n_fft, 1.)
    wsin = np.empty((freq_bins,1,n_fft))
    wcos = np.empty((freq_bins,1,n_fft))
    start_freq = low
    end_freq = high


    # num_cycles = start_freq*d/44000.
    # scaling_ind = np.log(end_freq/start_freq)/k

    if windowing=="no":
        window_mask = 1
    elif windowing=="hann":
        window_mask = 0.5-0.5*np.cos(2*np.pi*s/(n_fft),dtype=np.float32) # same as hann(n_fft, sym=False)
    else:
        raise Exception("Unknown windowing mode, please chooes either \"no\" or \"hann\"")

    if freq_scale == 'linear':
        start_bin = start_freq*n_fft/sr
        scaling_ind = (end_freq/start_freq)/freq_bins
        for k in range(freq_bins): # Only half of the bins contain useful info
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*(k*scaling_ind*start_bin)*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*(k*scaling_ind*start_bin)*s/n_fft)
    elif freq_scale == 'log':
        start_bin = start_freq*n_fft/sr
        scaling_ind = np.log(end_freq/start_freq)/freq_bins
        for k in range(freq_bins): # Only half of the bins contain useful info
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
    elif freq_scale == 'no':
        for k in range(freq_bins): # Only half of the bins contain useful info
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*k*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*k*s/n_fft)
    else:
        print("Please select the correct frequency scale, 'linear' or 'log'")
    return wsin.astype(np.float32),wcos.astype(np.float32)

def create_cqt_kernals(fs, fmin, fmax=None, n_bins=84, bins_per_octave=12, window='hann'):
    Q = 1/(2**(1/bins_per_octave)-1)
    fftLen = 2**nextpow2(np.ceil(Q * fs / fmin))
    # minWin = 2**nextpow2(np.ceil(Q * fs / fmax))
    if (fmax != None) and  (n_bins == None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin)) # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))    
    elif (fmax == None) and  (n_bins != None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    else:
        warnings.warn('If fmax is given, n_bins will be ignored',SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin)) # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)    
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = np.ceil(Q * fs / freq)
        # Centering the kernals
        if l%2==1: # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))-1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))
        tempKernel[k, start:start + int(l)] = get_window(window,int(l), fftbins=True)*np.exp(np.r_[-l//2:l//2]*1j*2*np.pi*freq/fs)/l
        # specKernel[k, :]=fft(conj(tempKernel[k, :]))
        specKernel[k, :] = fft(tempKernel[k])
        
    return specKernel[:,:fftLen//2+1], fftLen

class CQT(torch.nn.Module):
    def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=False, pad_mode='reflect'):
        super(CQT, self).__init__()
        #To Do center = False#
        
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        
        # creating kernals for CQT
        self.cqt_kernals, self.kernal_width = create_cqt_kernals(sr, fmin, fmax, n_bins, bins_per_octave, window)
        self.cqt_kernals_real = torch.tensor(self.cqt_kernals.real)
        self.cqt_kernals_imag = torch.tensor(self.cqt_kernals.imag)
        print("before norm", self.cqt_kernals_real.max())
        if self.norm: # Normalizing the filter
            torch_complex = torch.stack((self.cqt_kernals_real, self.cqt_kernals_imag), dim=0) 
            # froming a complex number shape = (real/imag, bins, n_fft)
            norm_value = torch.norm(torch_complex, p=2, dim=0)
            self.cqt_kernals_real = torch_complex[0]/torch.norm(norm_value, p=self.norm, dim=0)
            self.cqt_kernals_imag = torch_complex[1]/torch.norm(norm_value, p=self.norm, dim=0)
            print("after norm", self.cqt_kernals_real.max())
        # creating kernals for stft
        wsin, wcos = create_fourier_kernals(self.kernal_width, windowing="no", freq_scale='no')
        self.wsin = torch.tensor(wsin)
        self.wcos = torch.tensor(wcos)        
        
    def forward(self,x):
        x = x[None, None, :]
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernal_width//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernal_width//2)
            print(x.shape)
            x = padding(x)
            print(x.shape)
        # STFT
        fourier_real = conv1d(x, self.wcos, stride=self.hop_length)
        fourier_imag = conv1d(x, self.wsin, stride=self.hop_length)
        
        # CQT
        CQT_real, CQT_imag = complex_mul((self.cqt_kernals_real, self.cqt_kernals_imag), 
                                         (fourier_real, fourier_imag))
        
        # Getting CQT Amplitude
        CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        
        return CQT