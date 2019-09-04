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
def create_fourier_kernals(n_fft, freq_bins=None, low=50,high=6000, sr=44100, freq_scale='linear', window='hann'):
    """
    If freq_scale is 'no', then low and high arguments will be ignored
    """
    if freq_bins==None:
        freq_bins = n_fft//2+1

    s = np.arange(0, n_fft, 1.)
    wsin = np.empty((freq_bins,1,n_fft))
    wcos = np.empty((freq_bins,1,n_fft))
    start_freq = low
    end_freq = high


    # num_cycles = start_freq*d/44000.
    # scaling_ind = np.log(end_freq/start_freq)/k

    # Choosing window shape

    window_mask = get_window(window,int(n_fft), fftbins=True)


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

def create_cqt_kernals(fs, fmin, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann'):
    # norm arg is not functioning
    
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
        lenghts = np.ceil(Q * fs / freqs)
        # Centering the kernals
        if l%2==1: # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))-1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))
        sig = get_window(window,int(l), fftbins=True)*np.exp(np.r_[-l//2:l//2]*1j*2*np.pi*freq/fs)/l
#         if norm: # Normalizing the filter # Trying to normalize like librosa
#             tempKernel[k, start:start + int(l)] = sig/np.linalg.norm(sig, norm)
#         else:
        tempKernel[k, start:start + int(l)] = sig
        # specKernel[k, :]=fft(conj(tempKernel[k, :]))
        specKernel[k, :] = fft(tempKernel[k])
        
    return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()

class CQT(torch.nn.Module):
    def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=False, pad_mode='reflect'):
        super(CQT, self).__init__()
        # norm arg is not functioning
        
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        
        # creating kernals for CQT
        self.cqt_kernals, self.kernal_width, lenghts = create_cqt_kernals(sr, fmin, fmax, n_bins, bins_per_octave, norm, window)
        self.cqt_kernals_real = torch.tensor(self.cqt_kernals.real)
        self.cqt_kernals_imag = torch.tensor(self.cqt_kernals.imag)

        # creating kernals for stft
#         self.cqt_kernals_real*=lenghts.unsqueeze(1)/self.kernal_width # Trying to normalize as librosa
#         self.cqt_kernals_imag*=lenghts.unsqueeze(1)/self.kernal_width
        wsin, wcos = create_fourier_kernals(self.kernal_width, window='ones', freq_scale='no')
        self.wsin = torch.tensor(wsin)
        self.wcos = torch.tensor(wcos)        
        
    def forward(self,x):
        x = x[None, None, :]
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernal_width//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernal_width//2)

            x = padding(x)

        # STFT
        fourier_real = conv1d(x, self.wcos, stride=self.hop_length)
        fourier_imag = conv1d(x, self.wsin, stride=self.hop_length)
        
        # CQT
        CQT_real, CQT_imag = complex_mul((self.cqt_kernals_real, self.cqt_kernals_imag), 
                                         (fourier_real, fourier_imag))
        
        # Getting CQT Amplitude
        CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        
        return CQT
    
class Spectrogram(torch.nn.Module):
    def __init__(self, n_fft, freq_bins=None, hop_length=None, window='hann', freq_scale='no', center=True, pad_mode='reflect', low=50,high=6000):
        super(Spectrogram, self).__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        
        # Create filter windows for stft
        wsin, wcos = create_fourier_kernals(n_fft, freq_bins=freq_bins, window=window, freq_scale=freq_scale, low=low,high=high)
        self.wsin = torch.tensor(wsin, dtype=torch.float)
        self.wcos = torch.tensor(wcos, dtype=torch.float)

    def forward(self,x):
        x = x[None, None, :]
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.n_fft//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.n_fft//2)

            x = padding(x)
        
        spec = conv1d(x, self.wsin, stride=self.stride).pow(2) \
           + conv1d(x, self.wcos, stride=self.stride).pow(2) # Doing STFT by using conv1d
        return spec
    