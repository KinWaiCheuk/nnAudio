import torch
import torch.nn as nn
from torch.nn.functional import conv1d, conv2d

import numpy as np
import torch
from time import time
import math
from scipy.signal import get_window
from scipy import signal
from scipy import fft
import warnings

from .librosa_filters import mel # Use it for PyPip
# from librosa_filters import mel # Use it for debug

sz_float = 4    # size of a float
epsilon = 10e-8 # fudge factor for normalization

# ---------------------------Filter design -----------------------------------
def create_lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.03):
    """
    calculate the highest frequency we need to preserve and the lowest frequency we allow to pass through. Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is Nyquist frequency of the signal BEFORE downsampling
    """ 
    
#     transitionBandwidth = 0.03 
    passbandMax = band_center / (1 + transitionBandwidth)
    stopbandMin = band_center * (1 + transitionBandwidth)

    # Unlike the filter tool we used online yesterday, this tool does
    # not allow us to specify how closely the filter matches our
    # specifications. Instead, we specify the length of the kernel.
    # The longer the kernel is, the more precisely it will match.
#     kernelLength = 256 

    # We specify a list of key frequencies for which we will require 
    # that the filter match a specific output gain.
    # From [0.0 to passbandMax] is the frequency range we want to keep
    # untouched and [stopbandMin, 1.0] is the range we want to remove
    keyFrequencies = [0.0, passbandMax, stopbandMin, 1.0]

    # We specify a list of output gains to correspond to the key
    # frequencies listed above.
    # The first two gains are 1.0 because they correspond to the first
    # two key frequencies. the second two are 0.0 because they 
    # correspond to the stopband frequencies
    gainAtKeyFrequencies = [1.0, 1.0, 0.0, 0.0]

    # This command produces the filter kernel coefficients
    filterKernel = signal.firwin2(kernelLength, keyFrequencies, gainAtKeyFrequencies)    
    
    return filterKernel.astype(np.float32)

def downsampling_by_n(x, filterKernel, n):
    """A helper function that downsamples the audio by a arbitary factor n. It is used in CQT2010 and CQT2010v2

    Parameters
    ----------
    x : torch.Tensor
        The input waveform in ``torch.Tensor`` type with shape ``(batch, 1, len_audio)`` 

    filterKernel : str
        Filter kernel in ``torch.Tensor`` type with shape ``(1, 1, len_kernel)``

    n : int
        The downsampling factor

    Returns
    -------
    torch.Tensor
        The downsampled waveform

    Examples
    --------
    >>> x_down = downsampling_by_n(x, filterKernel)
    """
    x = conv1d(x,filterKernel,stride=n, padding=(filterKernel.shape[-1]-1)//2)
    return x

def downsampling_by_2(x, filterKernel):
    """A helper function that downsamples the audio by half. It is used in CQT2010 and CQT2010v2

    Parameters
    ----------
    x : torch.Tensor
        The input waveform in ``torch.Tensor`` type with shape ``(batch, 1, len_audio)``

    filterKernel : str
        Filter kernel in ``torch.Tensor`` type with shape ``(1, 1, len_kernel)``

    Returns
    -------
    torch.Tensor
        The downsampled waveform

    Examples
    --------
    >>> x_down = downsampling_by_2(x, filterKernel)
    """
    x = conv1d(x,filterKernel,stride=2, padding=(filterKernel.shape[-1]-1)//2)
    return x


## Basic tools for computation ##
def nextpow2(A):
    """A helper function to calculate the next nearest number to the power of 2. 

    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2

    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``

    Examples
    --------

    >>> nextpow2(6)
    3
    """
    return int(np.ceil(np.log2(A)))

def complex_mul(cqt_filter, stft):
    """Since PyTorch does not support complex numbers and its operation. We need to write our own complex multiplication function. This one is specially designed for CQT usage

    Parameters
    ----------
    cqt_filter : tuple of torch.Tensor
        The tuple is in the format of ``(real_torch_tensor, imag_torch_tensor)``

    Returns
    -------
    tuple of torch.Tensor
        The output is in the format of ``(real_torch_tensor, imag_torch_tensor)``
    """    
    
    cqt_filter_real = cqt_filter[0]
    cqt_filter_imag = cqt_filter[1]
    fourier_real = stft[0]
    fourier_imag = stft[1]
    
    CQT_real = torch.matmul(cqt_filter_real, fourier_real) - torch.matmul(cqt_filter_imag, fourier_imag)
    CQT_imag = torch.matmul(cqt_filter_real, fourier_imag) + torch.matmul(cqt_filter_imag, fourier_real)   
    
    return CQT_real, CQT_imag

def broadcast_dim(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """    

    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError("Only support input with shape = (batch, len) or shape = (len)")        
    return x

def broadcast_dim_conv2d(x):
    """
    Auto broadcast input so that it can fits into a Conv2d
    """    
    if x.dim() == 3:
        x = x[:, None, :,:]

    else:
        raise ValueError("Only support input with shape = (batch, len) or shape = (len)")        
    return x


## Kernal generation functions ##
def create_fourier_kernels(n_fft, freq_bins=None, fmin=50,fmax=6000, sr=44100, freq_scale='linear', window='hann'):
    """ This function creates the Fourier Kernel for STFT, Melspectrogram and CQT. Most of the parameters follow librosa conventions. Part of the code comes from pytorch_musicnet. https://github.com/jthickstun/pytorch_musicnet

    Parameters
    ----------
    n_fft : int
        The window size  

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins

    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument does nothing.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.
    
    freq_scale: 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin. When 'linear' or 'log' is used, the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.

    Returns
    -------
    wsin : numpy.array
        Imaginary Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``
        
    wcos : numpy.array
        Real Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``
    
    bins2freq : list
        Mapping each frequency bin to frequency in Hz.

    binslist : list
        The normalized frequency ``k`` in digital domain. This ``k`` is in the Discrete Fourier Transform equation $$ 

    """    
    if freq_bins==None:
        freq_bins = n_fft//2+1

    s = np.arange(0, n_fft, 1.)
    wsin = np.empty((freq_bins,1,n_fft))
    wcos = np.empty((freq_bins,1,n_fft))
    start_freq = fmin
    end_freq = fmax
    bins2freq = []
    binslist = []
    
    # num_cycles = start_freq*d/44000.
    # scaling_ind = np.log(end_freq/start_freq)/k

    # Choosing window shape

    window_mask = get_window(window,int(n_fft), fftbins=True)


    if freq_scale == 'linear':
        print("sampling rate = {}. Please make sure the sampling rate is correct in order to get a valid freq range".format(sr))
        start_bin = start_freq*n_fft/sr
        scaling_ind = (end_freq-start_freq)*(n_fft/sr)/freq_bins
        for k in range(freq_bins): # Only half of the bins contain useful info
#             print("linear freq = {}".format((k*scaling_ind+start_bin)*sr/n_fft))
            bins2freq.append((k*scaling_ind+start_bin)*sr/n_fft)
            binslist.append((k*scaling_ind+start_bin))
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*(k*scaling_ind+start_bin)*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*(k*scaling_ind+start_bin)*s/n_fft)
            
    elif freq_scale == 'log':
        print("sampling rate = {}. Please make sure the sampling rate is correct in order to get a valid freq range".format(sr))
        start_bin = start_freq*n_fft/sr
        scaling_ind = np.log(end_freq/start_freq)/freq_bins
        for k in range(freq_bins): # Only half of the bins contain useful info
#             print("log freq = {}".format(np.exp(k*scaling_ind)*start_bin*sr/n_fft))
            bins2freq.append(np.exp(k*scaling_ind)*start_bin*sr/n_fft)
            binslist.append((np.exp(k*scaling_ind)*start_bin))
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
            
    elif freq_scale == 'no':
        for k in range(freq_bins): # Only half of the bins contain useful info
            bins2freq.append(k*sr/n_fft)
            binslist.append(k)
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*k*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*k*s/n_fft)
    else:
        print("Please select the correct frequency scale, 'linear' or 'log'")
    return wsin.astype(np.float32),wcos.astype(np.float32), bins2freq, binslist

def create_cqt_kernels(Q, fs, fmin, n_bins=84, bins_per_octave=12, norm=1, window='hann', fmax=None, topbin_check=True):
    """
    Automatically create CQT kernels and convert it to frequency domain
    """
    # norm arg is not functioning
    
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
    if np.max(freqs) > fs/2 and topbin_check==True:
        raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins'.format(np.max(freqs)))
    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)    
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = np.ceil(Q * fs / freq)
        lenghts = np.ceil(Q * fs / freqs)
        # Centering the kernels
        if l%2==1: # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))-1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))
        sig = get_window(window,int(l), fftbins=True)*np.exp(np.r_[-l//2:l//2]*1j*2*np.pi*freq/fs)/l
        if norm: # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start:start + int(l)] = sig/np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start:start + int(l)] = sig
#         specKernel[k, :] = fft(tempKernel[k])
        
#     return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()
    return tempKernel, fftLen, torch.tensor(lenghts).float()

def create_cqt_kernels_t(Q, fs, fmin, n_bins=84, bins_per_octave=12, norm=1, window='hann', fmax=None):
    """
    Create cqt kernels in time-domain
    """
    # norm arg is not functioning
    
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
    if np.max(freqs) > fs/2:
        raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins'.format(np.max(freqs)))
    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)    
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = np.ceil(Q * fs / freq)
        lenghts = np.ceil(Q * fs / freqs)
        # Centering the kernels
        if l%2==1: # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))-1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))
        sig = get_window(window,int(l), fftbins=True)*np.exp(np.r_[-l//2:l//2]*1j*2*np.pi*freq/fs)/l
        if norm: # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start:start + int(l)] = sig/np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start:start + int(l)] = sig
        # specKernel[k, :]=fft(conj(tempKernel[k, :]))
        
    return tempKernel, fftLen, torch.tensor(lenghts).float()


### ------------------Spectrogram Classes---------------------------###

class STFT(torch.nn.Module):
    """This function is to calculate the short-time Fourier transform (STFT) of the input signal. Input signal should be in either of the following shapes. 1. ``(len_audio)``, 2. ``(num_audio, len_audio)``, 3. ``(num_audio, 1, len_audio)``. The correct shape will be inferred autommatically if the input follows these 3 shapes. Most of the arguments follow the convention from librosa. This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    Parameters
    ----------
    n_fft : int
        The window size. Default value is 2048. 

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins
    
    hop_length : int
        The hop (or stride) size. Default value is 512.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to scipy documentation for possible windowing functions. The default value is 'hann'

    freq_scale : 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin. When `linear` or `log` is used, the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``, the time index is the beginning of the STFT kernel, if ``True``, the time index is the center of the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.
    
    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument does nothing.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.

    trainable : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT kernels will also be caluclated and the STFT kernels will be updated during model training. Default value is ``False``

    output_format : str
        Determine the return type. ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``; ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``; ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``. The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'. 

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    device : str
        Choose which device to initialize this layer. Default value is 'cuda:0'

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)`` if 'Magnitude' is used as the ``output_format``; Shape = ``(num_samples, freq_bins,time_steps, 2)`` if 'Complex' or 'Phase' are used as the ``output_format``

    Examples
    --------
    >>> spec_layer = Spectrogram.STFT()
    >>> specs = spec_layer(x)
    """
    def __init__(self, n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050, trainable=False, output_format='Magnitude', verbose=True, device='cuda:0'):
        self.trainable = trainable
        super(STFT, self).__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.trainable = trainable
        self.output_format=output_format
        self.device = device
        start = time()
        # Create filter windows for stft
        wsin, wcos, self.bins2freq, self.bin_list = create_fourier_kernels(n_fft, freq_bins=freq_bins, window=window, freq_scale=freq_scale, fmin=fmin,fmax=fmax, sr=sr)
        self.wsin = torch.tensor(wsin, dtype=torch.float, device=self.device)
        self.wcos = torch.tensor(wcos, dtype=torch.float, device=self.device)

        # Making all these variables nn.Parameter, so that the model can be used with nn.Parallel
        self.wsin = torch.nn.Parameter(self.wsin, requires_grad=self.trainable)
        self.wcos = torch.nn.Parameter(self.wcos, requires_grad=self.trainable)

        # if self.trainable==True:
        #     self.wsin = torch.nn.Parameter(self.wsin)
        #     self.wcos = torch.nn.Parameter(self.wcos)

        if verbose==True:
            print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))
        else:
            pass

    def forward(self,x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.n_fft//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.n_fft//2)

            x = padding(x)
            
        spec_imag = conv1d(x, self.wsin, stride=self.stride) 
        spec_real = conv1d(x, self.wcos, stride=self.stride) # Doing STFT by using conv1d
        
        if self.output_format=='Magnitude':
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable==True:
                return torch.sqrt(spec+1e-8) # prevent Nan gradient when sqrt(0) due to output=0
            else:
                return torch.sqrt(spec)
        elif self.output_format=='Complex':
            return torch.stack((spec_real,-spec_imag), -1) # Remember the minus sign for imaginary part
            
        elif self.output_format=='Phase':
            return torch.atan2(-spec_imag+0.0,spec_real) # +0.0 helps remove -0.0 elements, which leads to error in calcuating pahse

            # This part is for implementing the librosa.core.magphase
            # But it seems it is not useful
            # phase_real = torch.cos(torch.atan2(spec_imag,spec_real))
            # phase_imag = torch.sin(torch.atan2(spec_imag,spec_real))
            # return torch.stack((phase_real,phase_imag), -1)
    
    def manual_forward(self,x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.n_fft//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.n_fft//2)

            x = padding(x)
            
        imag = conv1d(x, self.wsin, stride=self.stride).pow(2)
        real = conv1d(x, self.wcos, stride=self.stride).pow(2) # Doing STFT by using conv1d
        return real, imag
    
class DFT(torch.nn.Module):
    """
    The inverse function only works for 1 single frame. i.e. input shape = (batch, n_fft, 1)
    """    
    def __init__(self, n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050):
        super(DFT, self).__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        
        # Create filter windows for stft
        wsin, wcos, self.bins2freq = create_fourier_kernels(n_fft, freq_bins=n_fft, window=window, freq_scale=freq_scale, fmin=fmin,fmax=fmax, sr=sr)
        self.wsin = torch.tensor(wsin, dtype=torch.float)
        self.wcos = torch.tensor(wcos, dtype=torch.float)        

    def forward(self,x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.n_fft//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.n_fft//2)

            x = padding(x)
            
        imag = conv1d(x, self.wsin, stride=self.stride)
        real = conv1d(x, self.wcos, stride=self.stride)
        return (real, -imag)   
    
    def inverse(self,x_real,x_imag):
        x_real = broadcast_dim(x_real)
        x_imag = broadcast_dim(x_imag)
        
        x_real.transpose_(1,2) # Prepare the right shape to do inverse
        x_imag.transpose_(1,2) # Prepare the right shape to do inverse
        
#         if self.center:
#             if self.pad_mode == 'constant':
#                 padding = nn.ConstantPad1d(self.n_fft//2, 0)
#             elif self.pad_mode == 'reflect':
#                 padding = nn.ReflectionPad1d(self.n_fft//2)

#             x_real = padding(x_real)
#             x_imag = padding(x_imag)
        
        # Watch out for the positive and negative signs
        #ifft = e^(+2\pi*j)*X
        
        #ifft(X_real) = (a1, a2)
        
        #ifft(X_imag)*1j = (b1, b2)*1j
        #                = (-b2, b1)
        
        a1 = conv1d(x_real, self.wcos, stride=self.stride)
        a2 = conv1d(x_real, self.wsin, stride=self.stride)
        b1 = conv1d(x_imag, self.wcos, stride=self.stride)
        b2 = conv1d(x_imag, self.wsin, stride=self.stride)     
                                                   
        imag = a2+b1
        real = a1-b2
        return (real/self.n_fft, imag/self.n_fft)    

class iSTFT_complex_2d(torch.nn.Module):
    def __init__(self, n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050):
        super(iSTFT_complex_2d, self).__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft

        # Create filter windows for stft
        wsin, wcos, self.bins2freq = create_fourier_kernels(n_fft, freq_bins=n_fft, window=window, freq_scale=freq_scale, fmin=fmin,fmax=fmax, sr=sr)
        self.wsin = torch.tensor(wsin, dtype=torch.float)
        self.wcos = torch.tensor(wcos, dtype=torch.float)
        
        self.wsin = self.wsin[:,:,:,None] #adjust the filter shape to fit into 2d Conv
        self.wcos = self.wcos[:,:,:,None]
        
    def forward(self,x_real,x_imag):
        x_real = broadcast_dim_conv2d(x_real)
        x_imag = broadcast_dim_conv2d(x_imag) # taking conjuate
        
        
#         if self.center:
#             if self.pad_mode == 'constant':
#                 padding = nn.ConstantPad1d(self.n_fft//2, 0)
#             elif self.pad_mode == 'reflect':
#                 padding = nn.ReflectionPad1d(self.n_fft//2)

#             x_real = padding(x_real)
#             x_imag = padding(x_imag)
        
        # Watch out for the positive and negative signs
        #ifft = e^(+2\pi*j)*X
        
        #ifft(X_real) = (a1, a2)
        
        #ifft(X_imag)*1j = (b1, b2)*1j
        #                = (-b2, b1)
        
        a1 = conv2d(x_real, self.wcos, stride=(1,1))
        a2 = conv2d(x_real, self.wsin, stride=(1,1))
        b1 = conv2d(x_imag, self.wcos, stride=(1,1))
        b2 = conv2d(x_imag, self.wsin, stride=(1,1))     
                                                   
        imag = a2+b1
        real = a1-b2
        return (real/self.n_fft, imag/self.n_fft)    
class MelSpectrogram(torch.nn.Module):
    """This function is to calculate the Melspectrogram of the input signal. Input signal should be in either of the following shapes. 1. ``(len_audio)``, 2. ``(num_audio, len_audio)``, 3. ``(num_audio, 1, len_audio)``. The correct shape will be inferred autommatically if the input follows these 3 shapes. Most of the arguments follow the convention from librosa. This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.

    n_fft : int
        The window size for the STFT. Default value is 2048

    n_mels : int
        The number of Mel filter banks. The filter banks maps the n_fft to mel bins. Default value is 128
    
    hop_length : int
        The hop (or stride) size. Default value is 512.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to scipy documentation for possible windowing functions. The default value is 'hann'

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``, the time index is the beginning of the STFT kernel, if ``True``, the time index is the center of the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    htk : bool
        When ``False`` is used, the Mel scale is quasi-logarithmic. When ``True`` is used, the Mel scale is logarithmic. The default value is ``False`` 
    
    fmin : int
        The starting frequency for the lowest Mel filter bank

    fmax : int
        The ending frequency for the highest Mel filter bank

    trainable_mel : bool
        Determine if the Mel filter banks are trainable or not. If ``True``, the gradients for Mel filter banks will also be caluclated and the Mel filter banks will be updated during model training. Default value is ``False``

    trainable_STFT : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT kernels will also be caluclated and the STFT kernels will be updated during model training. Default value is ``False``
    
    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    device : str
        Choose which device to initialize this layer. Default value is 'cuda:0'

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)``.

    Examples
    --------
    >>> spec_layer = Spectrogram.MelSpectrogram()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=22050, n_fft=2048, n_mels=128, hop_length=512, window='hann', center=True, pad_mode='reflect', power=2.0, htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False, verbose=True, device='cuda:0'):
        super(MelSpectrogram, self).__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.device = device
        self.power = power
        
        # Create filter windows for stft
        start = time()
        wsin, wcos, self.bins2freq, _ = create_fourier_kernels(n_fft, freq_bins=None, window=window, freq_scale='no', sr=sr)
        self.wsin = torch.tensor(wsin, dtype=torch.float, device=self.device)
        self.wcos = torch.tensor(wcos, dtype=torch.float, device=self.device)
        

        # Creating kenral for mel spectrogram
        start = time()
        mel_basis = mel(sr, n_fft, n_mels, fmin, fmax, htk=htk, norm=norm)
        self.mel_basis = torch.tensor(mel_basis, device=self.device)

        if verbose==True:
            print("STFT filter created, time used = {:.4f} seconds".format(time()-start))
            print("Mel filter created, time used = {:.4f} seconds".format(time()-start))
        else:
            pass
        # Making everything nn.Prarmeter, so that this model can support nn.DataParallel
        self.mel_basis = torch.nn.Parameter(self.mel_basis, requires_grad=trainable_mel)
        self.wsin = torch.nn.Parameter(self.wsin, requires_grad=trainable_STFT)
        self.wcos = torch.nn.Parameter(self.wcos, requires_grad=trainable_STFT)          

        # if trainable_mel==True:
        #     self.mel_basis = torch.nn.Parameter(self.mel_basis)
        # if trainable_STFT==True:
        #     self.wsin = torch.nn.Parameter(self.wsin)
        #     self.wcos = torch.nn.Parameter(self.wcos)            
        
    def forward(self,x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.n_fft//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.n_fft//2)

            x = padding(x)
        
        spec = torch.sqrt(conv1d(x, self.wsin, stride=self.stride).pow(2) \
           + conv1d(x, self.wcos, stride=self.stride).pow(2))**self.power # Doing STFT by using conv1d
        
        melspec = torch.matmul(self.mel_basis, spec)
        return melspec    
    
    
class MFCC(torch.nn.Module):
    """This function is to calculate the Mel-frequency cepstral coefficients (MFCCs) of the input signal. It only support type-II DCT at the moment. Input signal should be in either of the following shapes. 1. ``(len_audio)``, 2. ``(num_audio, len_audio)``, 3. ``(num_audio, 1, len_audio)``. The correct shape will be inferred autommatically if the input follows these 3 shapes. Most of the arguments follow the convention from librosa. This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.

    n_mfcc : int
        The number of Mel-frequency cepstral coefficients
        
    norm : string
        The default value is 'ortho'. Normalization for DCT basis
    
    **kwargs
        Other arguments for Melspectrogram such as n_fft, n_mels, hop_length, and window

    Returns
    -------
    MFCCs : torch.tensor
        It returns a tensor of MFCCs.  shape = ``(num_samples, n_mfcc, time_steps)``.

    Examples
    --------
    >>> spec_layer = Spectrogram.MFCC()
    >>> mfcc = spec_layer(x)
    """    
    
    
    def __init__(self, sr=22050, n_mfcc=20, norm='ortho', device='cuda:0', verbose=True, **kwargs):
        super(MFCC, self).__init__()
        self.melspec_layer = MelSpectrogram(sr=sr, verbose=verbose, device=device, **kwargs)
        self.p2d = self.power_to_db()
        self.m_mfcc = n_mfcc
        
    def forward(self, x):
        x = self.melspec_layer(x)
        x = self.p2d.forward(x)
        x = self.dct(x, norm='ortho')[:,:self.m_mfcc,:]
        return x
        
    class power_to_db():
        # refer to https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#power_to_db for the original implmentation
        def __init__(self, ref=1.0, amin=1e-10, top_db=80.0, device='cuda:0'):
            if amin <= 0:
                raise ParameterError('amin must be strictly positive')
            self.amin = torch.tensor([amin], device=device)
            self.ref = torch.abs(torch.tensor([ref], device=device))
            self.top_db = top_db

        def forward(self, S):
            log_spec = 10.0 * torch.log10(torch.max(S, self.amin))
            log_spec -= 10.0 * torch.log10(torch.max(self.amin, self.ref))
            if self.top_db is not None:
                if self.top_db < 0:
                    raise ParameterError('top_db must be non-negative')
                batch_wise_max = log_spec.flatten(1).max(1)[0].unsqueeze(1).unsqueeze(1) # make the dim same as log_spec so that it can be boardcaseted
                log_spec = torch.max(log_spec, batch_wise_max - self.top_db)       
            return log_spec
    
    def dct(self, x, norm=None):    
        # refer to https://github.com/zh217/torch-dct for the original implmentation
        x = x.permute(0,2,1) # make freq the last axis, since dct applies to the frequency axis
        x_shape = x.shape
        N = x_shape[-1]

        v = torch.cat([x[:, :, ::2], x[:, :, 1::2].flip([2])], dim=2)
        Vc = torch.rfft(v, 1, onesided=False)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, :, 0] * W_r - Vc[:, :, :, 1] * W_i

        if norm == 'ortho':
            V[:, :, 0] /= np.sqrt(N) * 2
            V[:, :, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V

        return V.permute(0,2,1) # swaping back the time axis and freq axis

class CQT1992(torch.nn.Module):
    def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect', device="cuda:0"):
        super(CQT1992, self).__init__()
        # norm arg is not functioning
        
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        self.device = device
        
        # creating kernels for CQT
        Q = 1/(2**(1/bins_per_octave)-1)
        
        print("Creating CQT kernels ...", end='\r')
        start = time()
        self.cqt_kernels, self.kernal_width, self.lenghts = create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)
        self.lenghts = self.lenghts.to(device)
        self.cqt_kernels = fft(self.cqt_kernels)[:,:self.kernal_width//2+1]
        self.cqt_kernels_real = torch.tensor(self.cqt_kernels.real.astype(np.float32), device=device)
        self.cqt_kernels_imag = torch.tensor(self.cqt_kernels.imag.astype(np.float32), device=device)
        print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
        
        # creating kernels for stft
#         self.cqt_kernels_real*=lenghts.unsqueeze(1)/self.kernal_width # Trying to normalize as librosa
#         self.cqt_kernels_imag*=lenghts.unsqueeze(1)/self.kernal_width
        print("Creating STFT kernels ...", end='\r')
        start = time()
        wsin, wcos, self.bins2freq, _ = create_fourier_kernels(self.kernal_width, window='ones', freq_scale='no')
        self.wsin = torch.tensor(wsin, device=device)
        self.wcos = torch.tensor(wcos, device=device)      
        print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))
        
    def forward(self,x):
        x = broadcast_dim(x)
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
        CQT_real, CQT_imag = complex_mul((self.cqt_kernels_real, self.cqt_kernels_imag), 
                                         (fourier_real, fourier_imag))
        
        # Getting CQT Amplitude
        CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        
        if self.norm:
            return CQT/self.kernal_width*torch.sqrt(self.lenghts.view(-1,1))
        else:
            return CQT*torch.sqrt(self.lenghts.view(-1,1))    
    

class CQT2010(torch.nn.Module):
    """
    This alogrithm is using the resampling method proposed in [1]. Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the input audio by a factor of 2 to convoluting it with the small CQT kernel. Everytime the input audio is downsampled, the CQT relative to the downsampled input is equavalent to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the code from the 1992 alogrithm [2] 
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992).
    
    early downsampling factor is to downsample the input audio to reduce the CQT kernel size. The result with and without early downsampling are more or less the same except in the very low frequency region where freq < 40Hz
    
    """
    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84, bins_per_octave=12, norm=True, basis_norm=1, window='hann', pad_mode='reflect', earlydownsample=True, verbose=True, device='cuda:0'):
        super(CQT2010, self).__init__()
        
        self.norm = norm # Now norm is used to normalize the final CQT result by dividing n_fft
        #basis_norm is for normlaizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins   
        self.earlydownsample = earlydownsample # We will activate eraly downsampling later if possible
        self.device = device
        
        Q = 1/(2**(1/bins_per_octave)-1) # It will be used to calculate filter_cutoff and creating CQT kernels
        
        # Creating lowpass filter and make it a torch tensor
        if verbose==True:
            print("Creating low pass filter ...", end='\r')
        start = time()
        self.lowpass_filter = torch.tensor( 
                                            create_lowpass_filter(
                                            band_center = 0.5, 
                                            kernelLength=256,
                                            transitionBandwidth=0.001), device=self.device)
        self.lowpass_filter = self.lowpass_filter[None,None,:] # Broadcast the tensor to the shape that fits conv1d
        if verbose==True:
            print("Low pass filter created, time used = {:.4f} seconds".format(time()-start))

        # Caluate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
#         print("n_octaves = ", self.n_octaves)
        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = fmin*2**(self.n_octaves-1)
        remainder = n_bins % bins_per_octave
#         print("remainder = ", remainder)
        if remainder==0:
            fmax_t = self.fmin_t*2**((bins_per_octave-1)/bins_per_octave) # Calculate the top bin frequency
        else:
            fmax_t = self.fmin_t*2**((remainder-1)/bins_per_octave) # Calculate the top bin frequency
        self.fmin_t = fmax_t/2**(1-1/bins_per_octave) # Adjusting the top minium bins
        if fmax_t > sr/2:
            raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins'.format(fmax_t))
            
        
        if self.earlydownsample == True: # Do early downsampling if this argument is True
            if verbose==True:
                print("Creating early downsampling filter ...", end='\r')
            start = time()            
            sr, self.hop_length, self.downsample_factor, self.early_downsample_filter, self.earlydownsample = self.get_early_downsample_params(sr, hop_length, fmax_t, Q, self.n_octaves, verbose)
            if verbose==True:
                print("Early downsampling filter created, time used = {:.4f} seconds".format(time()-start))
        else:
            self.downsample_factor=1.
        
        # Preparing CQT kernels
        if verbose==True:
            print("Creating CQT kernels ...", end='\r')
        start = time()
#         print("Q = {}, fmin_t = {}, n_filters = {}".format(Q, self.fmin_t, n_filters))
        basis, self.n_fft, _ = create_cqt_kernels(Q, sr, self.fmin_t, n_filters, bins_per_octave, norm=basis_norm, topbin_check=False)
    
        # This is for the normalization in the end
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        lenghts = np.ceil(Q * sr / freqs)
        self.lenghts = torch.tensor(lenghts, device=self.device).float()

    
        self.basis=basis
        fft_basis = fft(basis)[:,:self.n_fft//2+1] # Convert CQT kenral from time domain to freq domain

        self.cqt_kernels_real = torch.tensor(fft_basis.real.astype(np.float32), device=self.device) # These cqt_kernal is already in the frequency domain
        self.cqt_kernels_imag = torch.tensor(fft_basis.imag.astype(np.float32), device=self.device)
        if verbose==True:
            print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
#         print("Getting cqt kernel done, n_fft = ",self.n_fft)
        # Preparing kernels for Short-Time Fourier Transform (STFT)
        # We set the frequency range in the CQT filter instead of here.
        if verbose==True:
            print("Creating STFT kernels ...", end='\r')
        start = time()
        wsin, wcos, self.bins2freq, _ = create_fourier_kernels(self.n_fft, window='ones', freq_scale='no')  
        self.wsin = torch.tensor(wsin, device=self.device)
        self.wcos = torch.tensor(wcos, device=self.device) 
        if verbose==True:
            print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))
        
        
        
        # If center==True, the STFT window will be put in the middle, and paddings at the beginning and ending are required.
        if self.pad_mode == 'constant':
            self.padding = nn.ConstantPad1d(self.n_fft//2, 0)
        elif self.pad_mode == 'reflect':
            self.padding = nn.ReflectionPad1d(self.n_fft//2)
                
    
    def get_cqt(self,x,hop_length, padding):
        """Multiplying the STFT result with the cqt_kernal, check out the 1992 CQT paper [1] for how to multiple the STFT result with the CQT kernel
        [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992)."""
        
        # STFT, converting the audio input from time domain to frequency domain
        try:
            x = padding(x) # When center == True, we need padding at the beginning and ending
        except:
            print("padding with reflection mode might not be the best choice, try using constant padding")
        fourier_real = conv1d(x, self.wcos, stride=hop_length)
        fourier_imag = conv1d(x, self.wsin, stride=hop_length)
        
        # Multiplying input with the CQT kernel in freq domain
        CQT_real, CQT_imag = complex_mul((self.cqt_kernels_real, self.cqt_kernels_imag), 
                                         (fourier_real, fourier_imag))
        
        # Getting CQT Amplitude
        CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        
        return CQT

    
    def get_early_downsample_params(self, sr, hop_length, fmax_t, Q, n_octaves, verbose):
        window_bandwidth = 1.5 # for hann window
        filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)   
        sr, hop_length, downsample_factor=self.early_downsample(sr, hop_length, n_octaves, sr//2, filter_cutoff)
        if downsample_factor != 1:
            if verbose==True:
                print("Can do early downsample, factor = ", downsample_factor)
            earlydownsample=True
#             print("new sr = ", sr)
#             print("new hop_length = ", hop_length)
            early_downsample_filter = create_lowpass_filter(band_center=1/downsample_factor, kernelLength=256, transitionBandwidth=0.03)
            early_downsample_filter = torch.tensor(early_downsample_filter, device=self.device)[None, None, :]
        else:       
            if verbose==True:     
                print("No early downsampling is required, downsample_factor = ", downsample_factor)
            early_downsample_filter = None
            earlydownsample=False
        return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample    
    
    # The following two downsampling count functions are obtained from librosa CQT 
    # They are used to determine the number of pre resamplings if the starting and ending frequency are both in low frequency regions.
    def early_downsample_count(self, nyquist, filter_cutoff, hop_length, n_octaves):
        '''Compute the number of early downsampling operations'''

        downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist /
                                                       filter_cutoff)) - 1) - 1)
#         print("downsample_count1 = ", downsample_count1)
        num_twos = nextpow2(hop_length)
        downsample_count2 = max(0, num_twos - n_octaves + 1)
#         print("downsample_count2 = ",downsample_count2)

        return min(downsample_count1, downsample_count2)

    def early_downsample(self, sr, hop_length, n_octaves,
                           nyquist, filter_cutoff):
        '''Return new sampling rate and hop length after early dowansampling'''
        downsample_count = self.early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
#         print("downsample_count = ", downsample_count)
        downsample_factor = 2**(downsample_count)

        hop_length //= downsample_factor # Getting new hop_length
        new_sr = sr / float(downsample_factor) # Getting new sampling rate

        sr = new_sr

        return sr, hop_length, downsample_factor
    
    
    def forward(self,x):
        x = broadcast_dim(x)
        if self.earlydownsample==True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        CQT = self.get_cqt(x, hop, self.padding) #Getting the top octave CQT
        
        x_down = x # Preparing a new variable for downsampling
        for i in range(self.n_octaves-1):  
            hop = hop//2   
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            CQT1 = self.get_cqt(x_down, hop, self.padding)
            CQT = torch.cat((CQT1, CQT),1) #
        CQT = CQT[:,-self.n_bins:,:] #Removing unwanted top bins
        CQT = CQT*2**(self.n_octaves-1) #Normalizing signals with respect to n_fft

        CQT = CQT*self.downsample_factor/2**(self.n_octaves-1) # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it same mag as 1992
        
        if self.norm:
            return CQT/self.n_fft*torch.sqrt(self.lenghts.view(-1,1))
        else:
            return CQT*torch.sqrt(self.lenghts.view(-1,1))

class CQT1992v2(torch.nn.Module):
    """This function is to calculate the CQT of the input signal. Input signal should be in either of the following shapes. 1. ``(len_audio)``, 2. ``(num_audio, len_audio)``, 3. ``(num_audio, 1, len_audio)``. The correct shape will be inferred autommatically if the input follows these 3 shapes. Most of the arguments follow the convention from librosa. This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    This alogrithm uses the method proposed in [1]. I slightly modify it so that it runs faster than the original 1992 algorithm, that is why I call it version 2. 
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992).

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.
    
    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.
    
    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization. Default is ``1``, which is same as the normalization used in librosa. 

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to scipy documentation for possible windowing functions. The default value is 'hann'

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels will also be caluclated and the CQT kernels will be updated during model training. Default value is ``False``

     output_format : str
        Determine the return type. ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``; ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``; ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``. The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'. 

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    device : str
        Choose which device to initialize this layer. Default value is 'cuda:0'

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)`` if 'Magnitude' is used as the ``output_format``; Shape = ``(num_samples, freq_bins,time_steps, 2)`` if 'Complex' or 'Phase' are used as the ``output_format``

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT1992v2()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect', trainable=False, output_format='Magnitude', verbose=True, device='cuda:0'):
        super(CQT1992v2, self).__init__()
        # norm arg is not functioning
        self.trainable = trainable
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.output_format = output_format
        self.device = device
        
        
        # creating kernels for CQT
        Q = 1/(2**(1/bins_per_octave)-1)
        
        if verbose==True:
            print("Creating CQT kernels ...", end='\r')

        start = time()
        self.cqt_kernels, self.kernal_width, self.lenghts = create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)
        self.lenghts = self.lenghts.to(device)
        self.cqt_kernels_real = torch.tensor(self.cqt_kernels.real, device=self.device).unsqueeze(1)
        self.cqt_kernels_imag = torch.tensor(self.cqt_kernels.imag, device=self.device).unsqueeze(1)
        
        # Making everything a Parameter to support nn.DataParallel    
        self.cqt_kernels_real = torch.nn.Parameter(self.cqt_kernels_real, requires_grad=trainable)
        self.cqt_kernels_imag = torch.nn.Parameter(self.cqt_kernels_imag, requires_grad=trainable) 
        self.lenghts = torch.nn.Parameter(self.lenghts, requires_grad=False)
        # if trainable==True:
        #     self.cqt_kernels_real = torch.nn.Parameter(self.cqt_kernels_real)
        #     self.cqt_kernels_imag = torch.nn.Parameter(self.cqt_kernels_imag)  
        
        if verbose==True:
            print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
        
        # creating kernels for stft
#         self.cqt_kernels_real*=lenghts.unsqueeze(1)/self.kernal_width # Trying to normalize as librosa
#         self.cqt_kernels_imag*=lenghts.unsqueeze(1)/self.kernal_width
        
    def forward(self,x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernal_width//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernal_width//2)

            x = padding(x)

        # CQT
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)*torch.sqrt(self.lenghts.view(-1,1))
        CQT_imag = -conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)*torch.sqrt(self.lenghts.view(-1,1))
        
        if self.output_format=='Magnitude':
            if self.trainable==False:
                # Getting CQT Amplitude
                CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
            else:
                CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2)+1e-8)              
            return CQT

        elif self.output_format=='Complex':
            return torch.stack((CQT_real,CQT_imag),-1)
        
        elif self.output_format=='Phase':
            phase_real = torch.cos(torch.atan2(CQT_imag,CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag,CQT_real))
            return torch.stack((phase_real,phase_imag), -1)  
        
    def forward_manual(self,x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernal_width//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernal_width//2)

            x = padding(x)

        # CQT
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)
        
        
        # Getting CQT Amplitude
        CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        return CQT*torch.sqrt(self.lenghts.view(-1,1))
        

class CQT2010v2(torch.nn.Module):
    """This function is to calculate the CQT of the input signal. Input signal should be in either of the following shapes. 1. ``(len_audio)``, 2. ``(num_audio, len_audio)``, 3. ``(num_audio, 1, len_audio)``. The correct shape will be inferred autommatically if the input follows these 3 shapes. Most of the arguments follow the convention from librosa. This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    This alogrithm uses the resampling method proposed in [1]. Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the input audio by a factor of 2 to convoluting it with the small CQT kernel. Everytime the input audio is downsampled, the CQT relative to the downsampled input is equavalent to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the code from the 1992 alogrithm [2] 
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992).
    
    early downsampling factor is to downsample the input audio to reduce the CQT kernel size. The result with and without early downsampling are more or less the same except in the very low frequency region where freq < 40Hz    

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.
    
    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.
    
    norm : bool
        Normalization for the CQT result.

    basis_norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization. Default is ``1``, which is same as the normalization used in librosa. 

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to scipy documentation for possible windowing functions. The default value is 'hann'

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels will also be caluclated and the CQT kernels will be updated during model training. Default value is ``False``

     output_format : str
        Determine the return type. 'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``; 'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``; 'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``. The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'. 

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    device : str
        Choose which device to initialize this layer. Default value is 'cuda:0'

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)`` if 'Magnitude' is used as the ``output_format``; Shape = ``(num_samples, freq_bins,time_steps, 2)`` if 'Complex' or 'Phase' are used as the ``output_format``

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT2010v2()
    >>> specs = spec_layer(x)
    """


    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84, bins_per_octave=12, norm=True, basis_norm=1, window='hann', pad_mode='reflect', earlydownsample=True, trainable=False, output_format='Magnitude', verbose=True, device='cuda:0'):
        super(CQT2010v2, self).__init__()
        
        self.norm = norm # Now norm is used to normalize the final CQT result by dividing n_fft
        #basis_norm is for normlaizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins   
        self.earlydownsample = earlydownsample # We will activate eraly downsampling later if possible
        self.trainable = trainable
        self.output_format = output_format
        self.device = device
        
        Q = 1/(2**(1/bins_per_octave)-1) # It will be used to calculate filter_cutoff and creating CQT kernels
        
        # Creating lowpass filter and make it a torch tensor
        if verbose==True:
            print("Creating low pass filter ...", end='\r')
        start = time()
#         self.lowpass_filter = torch.tensor( 
#                                             create_lowpass_filter(
#                                             band_center = 0.50, 
#                                             kernelLength=256,
#                                             transitionBandwidth=0.001))        
        self.lowpass_filter = torch.tensor( 
                                            create_lowpass_filter(
                                            band_center = 0.50, 
                                            kernelLength=256,
                                            transitionBandwidth=0.001), device=self.device)
        self.lowpass_filter = self.lowpass_filter[None,None,:] # Broadcast the tensor to the shape that fits conv1d
        if verbose==True:
            print("Low pass filter created, time used = {:.4f} seconds".format(time()-start))

        # Caluate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        if verbose==True:
            print("num_octave = ", self.n_octaves)
        
        # Calculate the lowest frequency bin for the top octave kernel      
        self.fmin_t = fmin*2**(self.n_octaves-1)
        remainder = n_bins % bins_per_octave
#         print("remainder = ", remainder)
        if remainder==0:
            fmax_t = self.fmin_t*2**((bins_per_octave-1)/bins_per_octave) # Calculate the top bin frequency
        else:
            fmax_t = self.fmin_t*2**((remainder-1)/bins_per_octave) # Calculate the top bin frequency
        self.fmin_t = fmax_t/2**(1-1/bins_per_octave) # Adjusting the top minium bins
        if fmax_t > sr/2:
            raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins'.format(fmax_t))        
        
        if self.earlydownsample == True: # Do early downsampling if this argument is True
            if verbose==True:
                print("Creating early downsampling filter ...", end='\r')
            start = time()            
            sr, self.hop_length, self.downsample_factor, self.early_downsample_filter, self.earlydownsample = self.get_early_downsample_params(sr, hop_length, fmax_t, Q, self.n_octaves, verbose)
            if verbose==True:
                print("Early downsampling filter created, time used = {:.4f} seconds".format(time()-start))
        else:
            self.downsample_factor=1.
        
        # Preparing CQT kernels
        if verbose==True:
            print("Creating CQT kernels ...", end='\r')
        start = time()
        basis, self.n_fft, self.lenghts = create_cqt_kernels(Q, sr, self.fmin_t, n_filters, bins_per_octave, norm=basis_norm, topbin_check=False)
        
        # For normalization in the end
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        lenghts = np.ceil(Q * sr / freqs)
        self.lenghts = torch.tensor(lenghts,device=self.device).float()        
        
        self.basis = basis
        self.cqt_kernels_real = torch.tensor(basis.real.astype(np.float32),device=self.device).unsqueeze(1) # These cqt_kernal is already in the frequency domain
        self.cqt_kernels_imag = torch.tensor(basis.imag.astype(np.float32),device=self.device).unsqueeze(1)

        # Making them nn.Parameter so that the model can support nn.DataParallel
        self.cqt_kernels_real = torch.nn.Parameter(self.cqt_kernels_real,  requires_grad=self.trainable)
        self.cqt_kernels_imag = torch.nn.Parameter(self.cqt_kernels_imag,  requires_grad=self.trainable) 
        self.lenghts = torch.nn.Parameter(self.lenghts, requires_grad=False)
        self.lowpass_filter = torch.nn.Parameter(self.lowpass_filter, requires_grad=False)
        # if trainable==True:
        #     self.cqt_kernels_real = torch.nn.Parameter(self.cqt_kernels_real)
        #     self.cqt_kernels_imag = torch.nn.Parameter(self.cqt_kernels_imag)          
        if verbose==True:
            print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
#         print("Getting cqt kernel done, n_fft = ",self.n_fft)      
        
        # If center==True, the STFT window will be put in the middle, and paddings at the beginning and ending are required.
        if self.pad_mode == 'constant':
            self.padding = nn.ConstantPad1d(self.n_fft//2, 0)
        elif self.pad_mode == 'reflect':
            self.padding = nn.ReflectionPad1d(self.n_fft//2)
                
    
    def get_cqt(self,x,hop_length, padding):
        """Multiplying the STFT result with the cqt_kernal, check out the 1992 CQT paper [1] for how to multiple the STFT result with the CQT kernel
        [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992)."""
        
        # STFT, converting the audio input from time domain to frequency domain
        try:
            x = padding(x) # When center == True, we need padding at the beginning and ending
        except:
            print("padding with reflection mode might not be the best choice, try using constant padding")
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=hop_length)
        CQT_imag = conv1d(x, self.cqt_kernels_imag, stride=hop_length)   
        
        # Getting CQT Amplitude
        CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        
        return CQT

    def get_cqt_complex(self,x,hop_length, padding):
        """Multiplying the STFT result with the cqt_kernal, check out the 1992 CQT paper [1] for how to multiple the STFT result with the CQT kernel
        [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992)."""
        
        # STFT, converting the audio input from time domain to frequency domain
        try:
            x = padding(x) # When center == True, we need padding at the beginning and ending
        except:
            print("padding with reflection mode might not be the best choice, try using constant padding")
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=hop_length)
        CQT_imag = -conv1d(x, self.cqt_kernels_imag, stride=hop_length)   
  
        return torch.stack((CQT_real, CQT_imag),-1)    
    
    def get_early_downsample_params(self, sr, hop_length, fmax_t, Q, n_octaves, verbose):
        window_bandwidth = 1.5 # for hann window
        filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)   
        sr, hop_length, downsample_factor=self.early_downsample(sr, hop_length, n_octaves, sr//2, filter_cutoff)
        if downsample_factor != 1:
            if verbose==True:
                print("Can do early downsample, factor = ", downsample_factor)
            earlydownsample=True
#             print("new sr = ", sr)
#             print("new hop_length = ", hop_length)
            early_downsample_filter = create_lowpass_filter(band_center=1/downsample_factor, kernelLength=256, transitionBandwidth=0.03)
            early_downsample_filter = torch.tensor(early_downsample_filter, device=self.device)[None, None, :]
        else:           
            if verbose==True: 
                print("No early downsampling is required, downsample_factor = ", downsample_factor)
            early_downsample_filter = None
            earlydownsample=False
        return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample    
    
    # The following two downsampling count functions are obtained from librosa CQT 
    # They are used to determine the number of pre resamplings if the starting and ending frequency are both in low frequency regions.
    def early_downsample_count(self, nyquist, filter_cutoff, hop_length, n_octaves):
        '''Compute the number of early downsampling operations'''

        downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist /
                                                       filter_cutoff)) - 1) - 1)
#         print("downsample_count1 = ", downsample_count1)
        num_twos = nextpow2(hop_length)
        downsample_count2 = max(0, num_twos - n_octaves + 1)
#         print("downsample_count2 = ",downsample_count2)

        return min(downsample_count1, downsample_count2)

    def early_downsample(self, sr, hop_length, n_octaves,
                           nyquist, filter_cutoff):
        '''Return new sampling rate and hop length after early dowansampling'''
        downsample_count = self.early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
#         print("downsample_count = ", downsample_count)
        downsample_factor = 2**(downsample_count)

        hop_length //= downsample_factor # Getting new hop_length
        new_sr = sr / float(downsample_factor) # Getting new sampling rate

        sr = new_sr

        return sr, hop_length, downsample_factor
    
    
    def forward(self,x):
        x = broadcast_dim(x)
        if self.earlydownsample==True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        CQT = self.get_cqt_complex(x, hop, self.padding) #Getting the top octave CQT
        
        x_down = x # Preparing a new variable for downsampling
        for i in range(self.n_octaves-1):  
            hop = hop//2   
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            CQT1 = self.get_cqt_complex(x_down, hop, self.padding)
            CQT = torch.cat((CQT1, CQT),1) #
        CQT = CQT[:,-self.n_bins:,:] #Removing unwanted bottom bins
        CQT = CQT*2**(self.n_octaves-1) #Normalizing signals with respect to n_fft
        # print("downsample_factor = ",self.downsample_factor)
        # print(CQT.shape)
        # print(self.lenghts.view(-1,1).shape)
        CQT = CQT*self.downsample_factor/2**(self.n_octaves-1) # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it same mag as 1992
        CQT = CQT*torch.sqrt(self.lenghts.view(-1,1,1)) # Normalize again to get same result as librosa
        
        if self.output_format=='Magnitude':
            if self.trainable==False:
                # Getting CQT Amplitude
                return torch.sqrt(CQT.pow(2).sum(-1))
            else:
                return torch.sqrt(CQT.pow(2).sum(-1)+1e-8)            

        elif self.output_format=='Complex':
            return CQT
        
        elif self.output_format=='Phase':
            phase_real = torch.cos(torch.atan2(CQT[:,:,:,1],CQT[:,:,:,0]))
            phase_imag = torch.sin(torch.atan2(CQT[:,:,:,1],CQT[:,:,:,0]))
            return torch.stack((phase_real,phase_imag), -1)  
        
    def forward_manual(self,x):
        x = broadcast_dim(x)
        if self.earlydownsample==True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        CQT = self.get_cqt(x, hop, self.padding) #Getting the top octave CQT
        
        x_down = x # Preparing a new variable for downsampling
        for i in range(self.n_octaves-1):  
            hop = hop//2   
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            CQT1 = self.get_cqt(x_down, hop, self.padding)
            CQT = torch.cat((CQT1, CQT),1) #
        CQT = CQT[:,-self.n_bins:,:] #Removing unwanted bottom bins
        CQT = CQT*2**(self.n_octaves-1) #Normalizing signals with respect to n_fft
        # print("downsample_factor = ",self.downsample_factor)
        CQT = CQT*self.downsample_factor/2**(self.n_octaves-1) # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it same mag as 1992

        return CQT*torch.sqrt(self.lenghts.view(-1,1))    
        
class CQT(CQT1992v2):
    """An abbreviation for CQT1992v2. Please refer to the CQT1992v2 documentation"""
    pass