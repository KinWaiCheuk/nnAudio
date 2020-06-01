# import torch
# import torch.nn as nn
# from torch.nn.functional import conv1d, conv2d
# # import librosa_filters.py
# import numpy as np
# import torch
# from time import time
# import math
# from scipy.signal import get_window
# from scipy import signal
# from scipy import fft
# import warnings

# sz_float = 4    # size of a float
# epsilon = 10e-8 # fudge factor for normalization

# # ---------------------------Filter design -----------------------------------
# def create_lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.03):
#     # calculate the highest frequency we need to preserve and the
#     # lowest frequency we allow to pass through. Note that frequency
#     # is on a scale from 0 to 1 where 0 is 0 and 1 is Nyquist 
#     # frequency of the signal BEFORE downsampling
    
# #     transitionBandwidth = 0.03 
#     passbandMax = band_center / (1 + transitionBandwidth)
#     stopbandMin = band_center * (1 + transitionBandwidth)

#     # Unlike the filter tool we used online yesterday, this tool does
#     # not allow us to specify how closely the filter matches our
#     # specifications. Instead, we specify the length of the kernel.
#     # The longer the kernel is, the more precisely it will match.
# #     kernelLength = 256 

#     # We specify a list of key frequencies for which we will require 
#     # that the filter match a specific output gain.
#     # From [0.0 to passbandMax] is the frequency range we want to keep
#     # untouched and [stopbandMin, 1.0] is the range we want to remove
#     keyFrequencies = [0.0, passbandMax, stopbandMin, 1.0]

#     # We specify a list of output gains to correspond to the key
#     # frequencies listed above.
#     # The first two gains are 1.0 because they correspond to the first
#     # two key frequencies. the second two are 0.0 because they 
#     # correspond to the stopband frequencies
#     gainAtKeyFrequencies = [1.0, 1.0, 0.0, 0.0]

#     # This command produces the filter kernel coefficients
#     filterKernel = signal.firwin2(kernelLength, keyFrequencies, gainAtKeyFrequencies)    
    
#     return filterKernel.astype(np.float32)

# def downsampling_by_n(x, filterKernel, n):
#     """downsampling by n"""
#     x = conv1d(x,filterKernel,stride=n, padding=(filterKernel.shape[-1]-1)//2)
#     return x

# def downsampling_by_2(x, filterKernel):
#     x = conv1d(x,filterKernel,stride=2, padding=(filterKernel.shape[-1]-1)//2)
#     return x


# ## Basic tools for computation ##
# def nextpow2(A):
#     return int(np.ceil(np.log2(A)))

# def complex_mul(cqt_filter, stft):
#     """Since PyTorch does not support complex numbers and its operation. We need to write our own complex multiplication function. This one is specially designed for CQT usage"""
    
#     cqt_filter_real = cqt_filter[0]
#     cqt_filter_imag = cqt_filter[1]
#     fourier_real = stft[0]
#     fourier_imag = stft[1]
    
#     CQT_real = torch.matmul(cqt_filter_real, fourier_real) - torch.matmul(cqt_filter_imag, fourier_imag)
#     CQT_imag = torch.matmul(cqt_filter_real, fourier_imag) + torch.matmul(cqt_filter_imag, fourier_real)   
    
#     return CQT_real, CQT_imag

# def broadcast_dim(x):
#     """
#     Auto broadcast input so that it can fits into a Conv1d
#     """
#     if x.dim() == 2:
#         x = x[:, None, :]
#     elif x.dim() == 1:
#         x = x[None, None, :]
#     elif x.dim() == 3:
#         pass
#     else:
#         raise ValueError("Only support input with shape = (batch, len) or shape = (len)")        
#     return x

# def broadcast_dim_conv2d(x):
#     """
#     To auto broadcast input so that it can fits into a Conv1d
#     """
#     if x.dim() == 3:
#         x = x[:, None, :,:]

#     else:
#         raise ValueError("Only support input with shape = (batch, len) or shape = (len)")        
#     return x


# ## Kernal generation functions ##
# def create_fourier_kernels(n_fft, freq_bins=None, fmin=50,fmax=6000, sr=44100, freq_scale='linear', window='hann'):
#     """
#     If freq_scale is 'no', then low and high arguments will be ignored
#     """
    
#     if freq_bins==None:
#         freq_bins = n_fft//2+1

#     s = np.arange(0, n_fft, 1.)
#     wsin = np.empty((freq_bins,1,n_fft))
#     wcos = np.empty((freq_bins,1,n_fft))
#     start_freq = fmin
#     end_freq = fmax
#     bins2freq = []
#     binslist = []
    
#     # num_cycles = start_freq*d/44000.
#     # scaling_ind = np.log(end_freq/start_freq)/k

#     # Choosing window shape

#     window_mask = get_window(window,int(n_fft), fftbins=True)


#     if freq_scale == 'linear':
#         print("sampling rate = {}. Please make sure the sampling rate is correct in order to get a valid freq range".format(sr))
#         start_bin = start_freq*n_fft/sr
#         scaling_ind = (end_freq-start_freq)*(n_fft/sr)/freq_bins
#         for k in range(freq_bins): # Only half of the bins contain useful info
# #             print("linear freq = {}".format((k*scaling_ind+start_bin)*sr/n_fft))
#             bins2freq.append((k*scaling_ind+start_bin)*sr/n_fft)
#             binslist.append((k*scaling_ind+start_bin))
#             wsin[k,0,:] = window_mask*np.sin(2*np.pi*(k*scaling_ind+start_bin)*s/n_fft)
#             wcos[k,0,:] = window_mask*np.cos(2*np.pi*(k*scaling_ind+start_bin)*s/n_fft)
            
#     elif freq_scale == 'log':
#         print("sampling rate = {}. Please make sure the sampling rate is correct in order to get a valid freq range".format(sr))
#         start_bin = start_freq*n_fft/sr
#         scaling_ind = np.log(end_freq/start_freq)/freq_bins
#         for k in range(freq_bins): # Only half of the bins contain useful info
# #             print("log freq = {}".format(np.exp(k*scaling_ind)*start_bin*sr/n_fft))
#             bins2freq.append(np.exp(k*scaling_ind)*start_bin*sr/n_fft)
#             binslist.append((np.exp(k*scaling_ind)*start_bin))
#             wsin[k,0,:] = window_mask*np.sin(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
#             wcos[k,0,:] = window_mask*np.cos(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
            
#     elif freq_scale == 'no':
#         for k in range(freq_bins): # Only half of the bins contain useful info
#             bins2freq.append(k*sr/n_fft)
#             binslist.append(k)
#             wsin[k,0,:] = window_mask*np.sin(2*np.pi*k*s/n_fft)
#             wcos[k,0,:] = window_mask*np.cos(2*np.pi*k*s/n_fft)
#     else:
#         print("Please select the correct frequency scale, 'linear' or 'log'")
#     return wsin.astype(np.float32),wcos.astype(np.float32), bins2freq, binslist

# def create_cqt_kernels(Q, fs, fmin, n_bins=84, bins_per_octave=12, norm=1, window='hann', fmax=None, topbin_check=True):
#     """
#     Automatically create CQT kernels and convert it to frequency domain
#     """
#     # norm arg is not functioning
    
#     fftLen = 2**nextpow2(np.ceil(Q * fs / fmin))
#     # minWin = 2**nextpow2(np.ceil(Q * fs / fmax))
#     if (fmax != None) and  (n_bins == None):
#         n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin)) # Calculate the number of bins
#         freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))    
#     elif (fmax == None) and  (n_bins != None):
#         freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave)) 
#     else:
#         warnings.warn('If fmax is given, n_bins will be ignored',SyntaxWarning)
#         n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin)) # Calculate the number of bins
#         freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
#     if np.max(freqs) > fs/2 and topbin_check==True:
#         raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins'.format(np.max(freqs)))
#     tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
#     specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)    
#     for k in range(0, int(n_bins)):
#         freq = freqs[k]
#         l = np.ceil(Q * fs / freq)
#         lenghts = np.ceil(Q * fs / freqs)
#         # Centering the kernels
#         if l%2==1: # pad more zeros on RHS
#             start = int(np.ceil(fftLen / 2.0 - l / 2.0))-1
#         else:
#             start = int(np.ceil(fftLen / 2.0 - l / 2.0))
#         sig = get_window(window,int(l), fftbins=True)*np.exp(np.r_[-l//2:l//2]*1j*2*np.pi*freq/fs)/l
#         if norm: # Normalizing the filter # Trying to normalize like librosa
#             tempKernel[k, start:start + int(l)] = sig/np.linalg.norm(sig, norm)
#         else:
#             tempKernel[k, start:start + int(l)] = sig
# #         specKernel[k, :] = fft(tempKernel[k])
        
# #     return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()
#     return tempKernel, fftLen, torch.tensor(lenghts).float()

# def create_cqt_kernels_t(Q, fs, fmin, n_bins=84, bins_per_octave=12, norm=1, window='hann', fmax=None):
#     """
#     Create cqt kernels in time-domain
#     """
#     # norm arg is not functioning
    
#     fftLen = 2**nextpow2(np.ceil(Q * fs / fmin))
#     # minWin = 2**nextpow2(np.ceil(Q * fs / fmax))
#     if (fmax != None) and  (n_bins == None):
#         n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin)) # Calculate the number of bins
#         freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))    
#     elif (fmax == None) and  (n_bins != None):
#         freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave)) 
#     else:
#         warnings.warn('If fmax is given, n_bins will be ignored',SyntaxWarning)
#         n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin)) # Calculate the number of bins
#         freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
#     if np.max(freqs) > fs/2:
#         raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins'.format(np.max(freqs)))
#     tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
#     specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)    
#     for k in range(0, int(n_bins)):
#         freq = freqs[k]
#         l = np.ceil(Q * fs / freq)
#         lenghts = np.ceil(Q * fs / freqs)
#         # Centering the kernels
#         if l%2==1: # pad more zeros on RHS
#             start = int(np.ceil(fftLen / 2.0 - l / 2.0))-1
#         else:
#             start = int(np.ceil(fftLen / 2.0 - l / 2.0))
#         sig = get_window(window,int(l), fftbins=True)*np.exp(np.r_[-l//2:l//2]*1j*2*np.pi*freq/fs)/l
#         if norm: # Normalizing the filter # Trying to normalize like librosa
#             tempKernel[k, start:start + int(l)] = sig/np.linalg.norm(sig, norm)
#         else:
#             tempKernel[k, start:start + int(l)] = sig
#         # specKernel[k, :]=fft(conj(tempKernel[k, :]))
        
#     return tempKernel, fftLen, torch.tensor(lenghts).float()

# ### ----------------Functions for generating kenral for Mel Spectrogram------------ ###
# def mel_to_hz(mels, htk=False):
#     """Convert mel bin numbers to frequencies

#     Examples
#     --------
#     >>> librosa.mel_to_hz(3)
#     200.

#     >>> librosa.mel_to_hz([1,2,3,4,5])
#     array([  66.667,  133.333,  200.   ,  266.667,  333.333])

#     Parameters
#     ----------
#     mels          : np.ndarray [shape=(n,)], float
#         mel bins to convert
#     htk           : bool
#         use HTK formula instead of Slaney

#     Returns
#     -------
#     frequencies   : np.ndarray [shape=(n,)]
#         input mels in Hz

#     See Also
#     --------
#     hz_to_mel
#     """

#     mels = np.asanyarray(mels)

#     if htk:
#         return 700.0 * (10.0**(mels / 2595.0) - 1.0)

#     # Fill in the linear scale
#     f_min = 0.0
#     f_sp = 200.0 / 3
#     freqs = f_min + f_sp * mels

#     # And now the nonlinear scale
#     min_log_hz = 1000.0                         # beginning of log region (Hz)
#     min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
#     logstep = np.log(6.4) / 27.0                # step size for log region

#     if mels.ndim:
#         # If we have vector data, vectorize
#         log_t = (mels >= min_log_mel)
#         freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
#     elif mels >= min_log_mel:
#         # If we have scalar data, check directly
#         freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

#     return freqs

# def hz_to_mel(frequencies, htk=False):
#     """Convert Hz to Mels

#     Examples
#     --------
#     >>> librosa.hz_to_mel(60)
#     0.9
#     >>> librosa.hz_to_mel([110, 220, 440])
#     array([ 1.65,  3.3 ,  6.6 ])

#     Parameters
#     ----------
#     frequencies   : number or np.ndarray [shape=(n,)] , float
#         scalar or array of frequencies
#     htk           : bool
#         use HTK formula instead of Slaney

#     Returns
#     -------
#     mels        : number or np.ndarray [shape=(n,)]
#         input frequencies in Mels

#     See Also
#     --------
#     mel_to_hz
#     """

#     frequencies = np.asanyarray(frequencies)

#     if htk:
#         return 2595.0 * np.log10(1.0 + frequencies / 700.0)

#     # Fill in the linear part
#     f_min = 0.0
#     f_sp = 200.0 / 3

#     mels = (frequencies - f_min) / f_sp

#     # Fill in the log-scale part

#     min_log_hz = 1000.0                         # beginning of log region (Hz)
#     min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
#     logstep = np.log(6.4) / 27.0                # step size for log region

#     if frequencies.ndim:
#         # If we have array data, vectorize
#         log_t = (frequencies >= min_log_hz)
#         mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
#     elif frequencies >= min_log_hz:
#         # If we have scalar data, heck directly
#         mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

#     return mels

# def fft_frequencies(sr=22050, n_fft=2048):
#     '''Alternative implementation of `np.fft.fftfreq`

#     Parameters
#     ----------
#     sr : number > 0 [scalar]
#         Audio sampling rate

#     n_fft : int > 0 [scalar]
#         FFT window size


#     Returns
#     -------
#     freqs : np.ndarray [shape=(1 + n_fft/2,)]
#         Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`


#     Examples
#     --------
#     >>> librosa.fft_frequencies(sr=22050, n_fft=16)
#     array([     0.   ,   1378.125,   2756.25 ,   4134.375,
#              5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])

#     '''

#     return np.linspace(0,
#                        float(sr) / 2,
#                        int(1 + n_fft//2),
#                        endpoint=True)

# def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
#     """Compute an array of acoustic frequencies tuned to the mel scale.

#     The mel scale is a quasi-logarithmic function of acoustic frequency
#     designed such that perceptually similar pitch intervals (e.g. octaves)
#     appear equal in width over the full hearing range.

#     Because the definition of the mel scale is conditioned by a finite number
#     of subjective psychoaoustical experiments, several implementations coexist
#     in the audio signal processing literature [1]_. By default, librosa replicates
#     the behavior of the well-established MATLAB Auditory Toolbox of Slaney [2]_.
#     According to this default implementation,  the conversion from Hertz to mel is
#     linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
#     replicates the Hidden Markov Toolkit [3]_ (HTK) according to the following formula:

#     `mel = 2595.0 * np.log10(1.0 + f / 700.0).`

#     The choice of implementation is determined by the `htk` keyword argument: setting
#     `htk=False` leads to the Auditory toolbox implementation, whereas setting it `htk=True`
#     leads to the HTK implementation.

#     .. [1] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
#         In Proc. International Conference on Acoustics, Speech, and Signal Processing
#         (ICASSP), vol. 1, pp. 217-220, 1998.

#     .. [2] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
#         Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

#     .. [3] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
#         Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
#         The HTK book, version 3.4. Cambridge University, March 2009.


#     See Also
#     --------
#     hz_to_mel
#     mel_to_hz
#     librosa.feature.melspectrogram
#     librosa.feature.mfcc


#     Parameters
#     ----------
#     n_mels    : int > 0 [scalar]
#         Number of mel bins.

#     fmin      : float >= 0 [scalar]
#         Minimum frequency (Hz).

#     fmax      : float >= 0 [scalar]
#         Maximum frequency (Hz).

#     htk       : bool
#         If True, use HTK formula to convert Hz to mel.
#         Otherwise (False), use Slaney's Auditory Toolbox.

#     Returns
#     -------
#     bin_frequencies : ndarray [shape=(n_mels,)]
#         Vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
#         axis.

#     Examples
#     --------
#     >>> librosa.mel_frequencies(n_mels=40)
#     array([     0.   ,     85.317,    170.635,    255.952,
#               341.269,    426.586,    511.904,    597.221,
#               682.538,    767.855,    853.173,    938.49 ,
#              1024.856,   1119.114,   1222.042,   1334.436,
#              1457.167,   1591.187,   1737.532,   1897.337,
#              2071.84 ,   2262.393,   2470.47 ,   2697.686,
#              2945.799,   3216.731,   3512.582,   3835.643,
#              4188.417,   4573.636,   4994.285,   5453.621,
#              5955.205,   6502.92 ,   7101.009,   7754.107,
#              8467.272,   9246.028,  10096.408,  11025.   ])

#     """

#     # 'Center freqs' of mel bands - uniformly spaced between limits
#     min_mel = hz_to_mel(fmin, htk=htk)
#     max_mel = hz_to_mel(fmax, htk=htk)

#     mels = np.linspace(min_mel, max_mel, n_mels)

#     return mel_to_hz(mels, htk=htk)

# def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
#         norm=1, dtype=np.float32):
#     """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

#     Parameters
#     ----------
#     sr        : number > 0 [scalar]
#         sampling rate of the incoming signal

#     n_fft     : int > 0 [scalar]
#         number of FFT components

#     n_mels    : int > 0 [scalar]
#         number of Mel bands to generate

#     fmin      : float >= 0 [scalar]
#         lowest frequency (in Hz)

#     fmax      : float >= 0 [scalar]
#         highest frequency (in Hz).
#         If `None`, use `fmax = sr / 2.0`

#     htk       : bool [scalar]
#         use HTK formula instead of Slaney

#     norm : {None, 1, np.inf} [scalar]
#         if 1, divide the triangular mel weights by the width of the mel band
#         (area normalization).  Otherwise, leave all the triangles aiming for
#         a peak value of 1.0

#     dtype : np.dtype
#         The data type of the output basis.
#         By default, uses 32-bit (single-precision) floating point.

#     Returns
#     -------
#     M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
#         Mel transform matrix

#     Notes
#     -----
#     This function caches at level 10.

#     Examples
#     --------
#     >>> melfb = librosa.filters.mel(22050, 2048)
#     >>> melfb
#     array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
#            [ 0.   ,  0.   , ...,  0.   ,  0.   ],
#            ...,
#            [ 0.   ,  0.   , ...,  0.   ,  0.   ],
#            [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


#     Clip the maximum frequency to 8KHz

#     >>> librosa.filters.mel(22050, 2048, fmax=8000)
#     array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
#            [ 0.  ,  0.  , ...,  0.  ,  0.  ],
#            ...,
#            [ 0.  ,  0.  , ...,  0.  ,  0.  ],
#            [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


#     >>> import matplotlib.pyplot as plt
#     >>> plt.figure()
#     >>> librosa.display.specshow(melfb, x_axis='linear')
#     >>> plt.ylabel('Mel filter')
#     >>> plt.title('Mel filter bank')
#     >>> plt.colorbar()
#     >>> plt.tight_layout()
#     >>> plt.show()
#     """

#     if fmax is None:
#         fmax = float(sr) / 2

#     if norm is not None and norm != 1 and norm != np.inf:
#         raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

#     # Initialize the weights
#     n_mels = int(n_mels)
#     weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

#     # Center freqs of each FFT bin
#     fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

#     # 'Center freqs' of mel bands - uniformly spaced between limits
#     mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

#     fdiff = np.diff(mel_f)
#     ramps = np.subtract.outer(mel_f, fftfreqs)

#     for i in range(n_mels):
#         # lower and upper slopes for all bins
#         lower = -ramps[i] / fdiff[i]
#         upper = ramps[i+2] / fdiff[i+1]

#         # .. then intersect them with each other and zero
#         weights[i] = np.maximum(0, np.minimum(lower, upper))

#     if norm == 1:
#         # Slaney-style mel is scaled to be approx constant energy per channel
#         enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
#         weights *= enorm[:, np.newaxis]

#     # Only check weights if f_mel[0] is positive
#     if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
#         # This means we have an empty channel somewhere
#         warnings.warn('Empty filters detected in mel frequency basis. '
#                       'Some channels will produce empty responses. '
#                       'Try increasing your sampling rate (and fmax) or '
#                       'reducing n_mels.')

#     return weights
# ### ------------------End of Functions for generating kenral for Mel Spectrogram ----------------###


# ### ------------------Spectrogram Classes---------------------------###
# class CQT1992(torch.nn.Module):
#     def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect'):
#         super(CQT1992, self).__init__()
#         # norm arg is not functioning
        
#         self.hop_length = hop_length
#         self.center = center
#         self.pad_mode = pad_mode
#         self.norm = norm
        
#         # creating kernels for CQT
#         Q = 1/(2**(1/bins_per_octave)-1)
        
#         print("Creating CQT kernels ...", end='\r')
#         start = time()
#         self.cqt_kernels, self.kernal_width, lenghts = create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)
#         self.cqt_kernels = fft(self.cqt_kernels)[:,:self.kernal_width//2+1]
#         self.cqt_kernels_real = torch.tensor(self.cqt_kernels.real.astype(np.float32))
#         self.cqt_kernels_imag = torch.tensor(self.cqt_kernels.imag.astype(np.float32))
#         print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
        
#         # creating kernels for stft
# #         self.cqt_kernels_real*=lenghts.unsqueeze(1)/self.kernal_width # Trying to normalize as librosa
# #         self.cqt_kernels_imag*=lenghts.unsqueeze(1)/self.kernal_width
#         print("Creating STFT kernels ...", end='\r')
#         start = time()
#         wsin, wcos, self.bins2freq, _ = create_fourier_kernels(self.kernal_width, window='ones', freq_scale='no')
#         self.wsin = torch.tensor(wsin)
#         self.wcos = torch.tensor(wcos)      
#         print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))
        
#     def forward(self,x):
#         x = broadcast_dim(x)
#         if self.center:
#             if self.pad_mode == 'constant':
#                 padding = nn.ConstantPad1d(self.kernal_width//2, 0)
#             elif self.pad_mode == 'reflect':
#                 padding = nn.ReflectionPad1d(self.kernal_width//2)

#             x = padding(x)

#         # STFT
#         fourier_real = conv1d(x, self.wcos, stride=self.hop_length)
#         fourier_imag = conv1d(x, self.wsin, stride=self.hop_length)
        
#         # CQT
#         CQT_real, CQT_imag = complex_mul((self.cqt_kernels_real, self.cqt_kernels_imag), 
#                                          (fourier_real, fourier_imag))
        
#         # Getting CQT Amplitude
#         CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        
#         if self.norm:
#             return CQT/self.kernal_width
#         else:
#             return CQT

# class CQT1992v2(torch.nn.Module):
#     def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect'):
#         super(CQT1992v2, self).__init__()
#         # norm arg is not functioning
        
#         self.hop_length = hop_length
#         self.center = center
#         self.pad_mode = pad_mode
        
#         # creating kernels for CQT
#         Q = 1/(2**(1/bins_per_octave)-1)
        
#         print("Creating CQT kernels ...", end='\r')
#         start = time()
#         self.cqt_kernels, self.kernal_width, lenghts = create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)
#         self.cqt_kernels_real = torch.tensor(self.cqt_kernels.real).unsqueeze(1)
#         self.cqt_kernels_imag = torch.tensor(self.cqt_kernels.imag).unsqueeze(1)
#         print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
        
#         # creating kernels for stft
# #         self.cqt_kernels_real*=lenghts.unsqueeze(1)/self.kernal_width # Trying to normalize as librosa
# #         self.cqt_kernels_imag*=lenghts.unsqueeze(1)/self.kernal_width
        
#     def forward(self,x):
#         x = broadcast_dim(x)
#         if self.center:
#             if self.pad_mode == 'constant':
#                 padding = nn.ConstantPad1d(self.kernal_width//2, 0)
#             elif self.pad_mode == 'reflect':
#                 padding = nn.ReflectionPad1d(self.kernal_width//2)

#             x = padding(x)

#         # CQT
#         CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
#         CQT_imag = conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)
        
        
#         # Getting CQT Amplitude
#         CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
#         return CQT

# class STFT(torch.nn.Module):
#     """When using freq_scale, please set the correct sampling rate. The sampling rate is used to calucate the correct frequency"""
    
#     def __init__(self, n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050):
#         super(STFT, self).__init__()
#         self.stride = hop_length
#         self.center = center
#         self.pad_mode = pad_mode
#         self.n_fft = n_fft
        
#         # Create filter windows for stft
#         wsin, wcos, self.bins2freq, self.bin_list = create_fourier_kernels(n_fft, freq_bins=freq_bins, window=window, freq_scale=freq_scale, fmin=fmin,fmax=fmax, sr=sr)
#         self.wsin = torch.tensor(wsin, dtype=torch.float)
#         self.wcos = torch.tensor(wcos, dtype=torch.float)

#     def forward(self,x):
#         x = broadcast_dim(x)
#         if self.center:
#             if self.pad_mode == 'constant':
#                 padding = nn.ConstantPad1d(self.n_fft//2, 0)
#             elif self.pad_mode == 'reflect':
#                 padding = nn.ReflectionPad1d(self.n_fft//2)

#             x = padding(x)
            
#         spec = conv1d(x, self.wsin, stride=self.stride).pow(2) \
#            + conv1d(x, self.wcos, stride=self.stride).pow(2) # Doing STFT by using conv1d
#         return torch.sqrt(spec)
    
#     def manual_forward(self,x):
#         x = broadcast_dim(x)
#         if self.center:
#             if self.pad_mode == 'constant':
#                 padding = nn.ConstantPad1d(self.n_fft//2, 0)
#             elif self.pad_mode == 'reflect':
#                 padding = nn.ReflectionPad1d(self.n_fft//2)

#             x = padding(x)
            
#         imag = conv1d(x, self.wsin, stride=self.stride).pow(2)
#         real = conv1d(x, self.wcos, stride=self.stride).pow(2) # Doing STFT by using conv1d
#         return real, imag
    
# class DFT(torch.nn.Module):
#     """
#     The inverse function only works for 1 single frame. i.e. input shape = (batch, n_fft, 1)
#     """    
#     def __init__(self, n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050):
#         super(DFT, self).__init__()
#         self.stride = hop_length
#         self.center = center
#         self.pad_mode = pad_mode
#         self.n_fft = n_fft
        
#         # Create filter windows for stft
#         wsin, wcos, self.bins2freq = create_fourier_kernels(n_fft, freq_bins=n_fft, window=window, freq_scale=freq_scale, fmin=fmin,fmax=fmax, sr=sr)
#         self.wsin = torch.tensor(wsin, dtype=torch.float)
#         self.wcos = torch.tensor(wcos, dtype=torch.float)

#     def forward(self,x):
#         x = broadcast_dim(x)
#         if self.center:
#             if self.pad_mode == 'constant':
#                 padding = nn.ConstantPad1d(self.n_fft//2, 0)
#             elif self.pad_mode == 'reflect':
#                 padding = nn.ReflectionPad1d(self.n_fft//2)

#             x = padding(x)
            
#         imag = conv1d(x, self.wsin, stride=self.stride)
#         real = conv1d(x, self.wcos, stride=self.stride)
#         return (real, -imag)   
    
#     def inverse(self,x_real,x_imag):
#         x_real = broadcast_dim(x_real)
#         x_imag = broadcast_dim(x_imag)
        
#         x_real.transpose_(1,2) # Prepare the right shape to do inverse
#         x_imag.transpose_(1,2) # Prepare the right shape to do inverse
        
# #         if self.center:
# #             if self.pad_mode == 'constant':
# #                 padding = nn.ConstantPad1d(self.n_fft//2, 0)
# #             elif self.pad_mode == 'reflect':
# #                 padding = nn.ReflectionPad1d(self.n_fft//2)

# #             x_real = padding(x_real)
# #             x_imag = padding(x_imag)
        
#         # Watch out for the positive and negative signs
#         #ifft = e^(+2\pi*j)*X
        
#         #ifft(X_real) = (a1, a2)
        
#         #ifft(X_imag)*1j = (b1, b2)*1j
#         #                = (-b2, b1)
        
#         a1 = conv1d(x_real, self.wcos, stride=self.stride)
#         a2 = conv1d(x_real, self.wsin, stride=self.stride)
#         b1 = conv1d(x_imag, self.wcos, stride=self.stride)
#         b2 = conv1d(x_imag, self.wsin, stride=self.stride)     
                                                   
#         imag = a2+b1
#         real = a1-b2
#         return (real/self.n_fft, imag/self.n_fft)    

# class iSTFT_complex_2d(torch.nn.Module):
#     def __init__(self, n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050):
#         super(iSTFT_complex_2d, self).__init__()
#         self.stride = hop_length
#         self.center = center
#         self.pad_mode = pad_mode
#         self.n_fft = n_fft

#         # Create filter windows for stft
#         wsin, wcos, self.bins2freq = create_fourier_kernels(n_fft, freq_bins=n_fft, window=window, freq_scale=freq_scale, fmin=fmin,fmax=fmax, sr=sr)
#         self.wsin = torch.tensor(wsin, dtype=torch.float)
#         self.wcos = torch.tensor(wcos, dtype=torch.float)
        
#         self.wsin = self.wsin[:,:,:,None] #adjust the filter shape to fit into 2d Conv
#         self.wcos = self.wcos[:,:,:,None]
        
#     def forward(self,x_real,x_imag):
#         x_real = broadcast_dim_conv2d(x_real)
#         x_imag = broadcast_dim_conv2d(x_imag) # taking conjuate
        
        
# #         if self.center:
# #             if self.pad_mode == 'constant':
# #                 padding = nn.ConstantPad1d(self.n_fft//2, 0)
# #             elif self.pad_mode == 'reflect':
# #                 padding = nn.ReflectionPad1d(self.n_fft//2)

# #             x_real = padding(x_real)
# #             x_imag = padding(x_imag)
        
#         # Watch out for the positive and negative signs
#         #ifft = e^(+2\pi*j)*X
        
#         #ifft(X_real) = (a1, a2)
        
#         #ifft(X_imag)*1j = (b1, b2)*1j
#         #                = (-b2, b1)
        
#         a1 = conv2d(x_real, self.wcos, stride=(1,1))
#         a2 = conv2d(x_real, self.wsin, stride=(1,1))
#         b1 = conv2d(x_imag, self.wcos, stride=(1,1))
#         b2 = conv2d(x_imag, self.wsin, stride=(1,1))     
                                                   
#         imag = a2+b1
#         real = a1-b2
#         return (real/self.n_fft, imag/self.n_fft)    
    
# class MelSpectrogram(torch.nn.Module):
#     def __init__(self, sr=22050, n_fft=2048, n_mels=128, hop_length=512, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1):
#         super(MelSpectrogram, self).__init__()
#         self.stride = hop_length
#         self.center = center
#         self.pad_mode = pad_mode
#         self.n_fft = n_fft
        
#         # Create filter windows for stft
#         wsin, wcos, self.bins2freq, _ = create_fourier_kernels(n_fft, freq_bins=None, window=window, freq_scale='no', sr=sr)
#         self.wsin = torch.tensor(wsin, dtype=torch.float)
#         self.wcos = torch.tensor(wcos, dtype=torch.float)

#         # Creating kenral for mel spectrogram
#         mel_basis = mel(sr, n_fft, n_mels, fmin, fmax, htk=htk, norm=norm)
#         self.mel_basis = torch.tensor(mel_basis)
#     def forward(self,x):
#         x = broadcast_dim(x)
#         if self.center:
#             if self.pad_mode == 'constant':
#                 padding = nn.ConstantPad1d(self.n_fft//2, 0)
#             elif self.pad_mode == 'reflect':
#                 padding = nn.ReflectionPad1d(self.n_fft//2)

#             x = padding(x)
        
#         spec = conv1d(x, self.wsin, stride=self.stride).pow(2) \
#            + conv1d(x, self.wcos, stride=self.stride).pow(2) # Doing STFT by using conv1d
        
#         melspec = torch.matmul(self.mel_basis, spec)
#         return melspec
    
    
#     ### ----------------CQT 2010------------------------------------------------------- ###

# def cqt_filter_fft(sr, fmin, n_bins, bins_per_octave, tuning,
#                      filter_scale, norm, sparsity, hop_length=None,
#                      window='hann'):
#     '''Generate the frequency domain constant-Q filter basis.'''

#     basis, lengths = filters.constant_q(sr,
#                                         fmin=fmin,
#                                         n_bins=n_bins,
#                                         bins_per_octave=bins_per_octave,
#                                         tuning=tuning,
#                                         filter_scale=filter_scale,
#                                         norm=norm,
#                                         pad_fft=True,
#                                         window=window)

#     # Filters are padded up to the nearest integral power of 2
#     n_fft = basis.shape[1]

#     if (hop_length is not None and
#             n_fft < 2.0**(1 + np.ceil(np.log2(hop_length)))):

#         n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

#     # re-normalize bases with respect to the FFT window length
#     basis *= lengths[:, np.newaxis] / float(n_fft)

#     # FFT and retain only the non-negative frequencies
#     fft = get_fftlib()
#     fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft // 2)+1]

#     # sparsify the basis
#     fft_basis = util.sparsify_rows(fft_basis, quantile=sparsity)

#     return fft_basis, n_fft, lengths

# # from librosa import filters, get_fftlib, util

# class CQT2010(torch.nn.Module):
#     """
#     This alogrithm is using the resampling method proposed in [1]. Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the input audio by a factor of 2 to convoluting it with the small CQT kernel. Everytime the input audio is downsampled, the CQT relative to the downsampled input is equavalent to the next lower octave.
#     The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the code from the 1992 alogrithm [2] 
#     [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
#     [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992).
    
#     early downsampling factor is to downsample the input audio to reduce the CQT kernel size. The result with and without early downsampling are more or less the same except in the very low frequency region where freq < 40Hz
    
#     """
#     def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=True, basis_norm=1, window='hann', pad_mode='reflect', earlydownsample=True):
#         super(CQT2010, self).__init__()
        
#         self.norm = norm # Now norm is used to normalize the final CQT result by dividing n_fft
#         #basis_norm is for normlaizing basis
#         self.hop_length = hop_length
#         self.pad_mode = pad_mode
#         self.n_bins = n_bins   
#         self.earlydownsample = earlydownsample # We will activate eraly downsampling later if possible
        
#         Q = 1/(2**(1/bins_per_octave)-1) # It will be used to calculate filter_cutoff and creating CQT kernels
        
#         # Creating lowpass filter and make it a torch tensor
#         print("Creating low pass filter ...", end='\r')
#         start = time()
#         self.lowpass_filter = torch.tensor( 
#                                             create_lowpass_filter(
#                                             band_center = 0.5, 
#                                             kernelLength=256,
#                                             transitionBandwidth=0.001))
#         self.lowpass_filter = self.lowpass_filter[None,None,:] # Broadcast the tensor to the shape that fits conv1d
#         print("Low pass filter created, time used = {:.4f} seconds".format(time()-start))

#         # Caluate num of filter requires for the kernel
#         # n_octaves determines how many resampling requires for the CQT
#         n_filters = min(bins_per_octave, n_bins)
#         self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
# #         print("n_octaves = ", self.n_octaves)
#         # Calculate the lowest frequency bin for the top octave kernel
#         self.fmin_t = fmin*2**(self.n_octaves-1)
#         remainder = n_bins % bins_per_octave
# #         print("remainder = ", remainder)
#         if remainder==0:
#             fmax_t = self.fmin_t*2**((bins_per_octave-1)/bins_per_octave) # Calculate the top bin frequency
#         else:
#             fmax_t = self.fmin_t*2**((remainder-1)/bins_per_octave) # Calculate the top bin frequency
#         self.fmin_t = fmax_t/2**(1-1/bins_per_octave) # Adjusting the top minium bins
#         if fmax_t > sr/2:
#             raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins'.format(fmax_t))
            
        
#         if self.earlydownsample == True: # Do early downsampling if this argument is True
#             print("Creating early downsampling filter ...", end='\r')
#             start = time()            
#             sr, self.hop_length, self.downsample_factor, self.early_downsample_filter, self.earlydownsample = self.get_early_downsample_params(sr, hop_length, fmax_t, Q, self.n_octaves)
#             print("Early downsampling filter created, time used = {:.4f} seconds".format(time()-start))
#         else:
#             self.downsample_factor=1.
        
#         # Preparing CQT kernels
#         print("Creating CQT kernels ...", end='\r')
#         start = time()
# #         print("Q = {}, fmin_t = {}, n_filters = {}".format(Q, self.fmin_t, n_filters))
#         basis, self.n_fft, _ = create_cqt_kernels(Q, sr, self.fmin_t, n_filters, bins_per_octave, norm=basis_norm, topbin_check=False)
#         self.basis=basis
#         fft_basis = fft(basis)[:,:self.n_fft//2+1] # Convert CQT kenral from time domain to freq domain

#         self.cqt_kernels_real = torch.tensor(fft_basis.real.astype(np.float32)) # These cqt_kernal is already in the frequency domain
#         self.cqt_kernels_imag = torch.tensor(fft_basis.imag.astype(np.float32))
#         print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
# #         print("Getting cqt kernel done, n_fft = ",self.n_fft)
#         # Preparing kernels for Short-Time Fourier Transform (STFT)
#         # We set the frequency range in the CQT filter instead of here.
#         print("Creating STFT kernels ...", end='\r')
#         start = time()
#         wsin, wcos, self.bins2freq, _ = create_fourier_kernels(self.n_fft, window='ones', freq_scale='no')  
#         self.wsin = torch.tensor(wsin)
#         self.wcos = torch.tensor(wcos) 
#         print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))
        
        
        
#         # If center==True, the STFT window will be put in the middle, and paddings at the beginning and ending are required.
#         if self.pad_mode == 'constant':
#             self.padding = nn.ConstantPad1d(self.n_fft//2, 0)
#         elif self.pad_mode == 'reflect':
#             self.padding = nn.ReflectionPad1d(self.n_fft//2)
                
    
#     def get_cqt(self,x,hop_length, padding):
#         """Multiplying the STFT result with the cqt_kernal, check out the 1992 CQT paper [1] for how to multiple the STFT result with the CQT kernel
#         [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992)."""
        
#         # STFT, converting the audio input from time domain to frequency domain
#         try:
#             x = padding(x) # When center == True, we need padding at the beginning and ending
#         except:
#             print("padding with reflection mode might not be the best choice, try using constant padding")
#         fourier_real = conv1d(x, self.wcos, stride=hop_length)
#         fourier_imag = conv1d(x, self.wsin, stride=hop_length)
        
#         # Multiplying input with the CQT kernel in freq domain
#         CQT_real, CQT_imag = complex_mul((self.cqt_kernels_real, self.cqt_kernels_imag), 
#                                          (fourier_real, fourier_imag))
        
#         # Getting CQT Amplitude
#         CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        
#         return CQT

    
#     def get_early_downsample_params(self, sr, hop_length, fmax_t, Q, n_octaves):
#         window_bandwidth = 1.5 # for hann window
#         filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)   
#         sr, hop_length, downsample_factor=self.early_downsample(sr, hop_length, n_octaves, sr//2, filter_cutoff)
#         if downsample_factor != 1:
#             print("Can do early downsample, factor = ", downsample_factor)
#             earlydownsample=True
# #             print("new sr = ", sr)
# #             print("new hop_length = ", hop_length)
#             early_downsample_filter = create_lowpass_filter(band_center=1/downsample_factor, kernelLength=256, transitionBandwidth=0.03)
#             early_downsample_filter = torch.tensor(early_downsample_filter)[None, None, :]
#         else:            
#             print("No early downsampling is required, downsample_factor = ", downsample_factor)
#             early_downsample_filter = None
#             earlydownsample=False
#         return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample    
    
#     # The following two downsampling count functions are obtained from librosa CQT 
#     # They are used to determine the number of pre resamplings if the starting and ending frequency are both in low frequency regions.
#     def early_downsample_count(self, nyquist, filter_cutoff, hop_length, n_octaves):
#         '''Compute the number of early downsampling operations'''

#         downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist /
#                                                        filter_cutoff)) - 1) - 1)
# #         print("downsample_count1 = ", downsample_count1)
#         num_twos = nextpow2(hop_length)
#         downsample_count2 = max(0, num_twos - n_octaves + 1)
# #         print("downsample_count2 = ",downsample_count2)

#         return min(downsample_count1, downsample_count2)

#     def early_downsample(self, sr, hop_length, n_octaves,
#                            nyquist, filter_cutoff):
#         '''Return new sampling rate and hop length after early dowansampling'''
#         downsample_count = self.early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
# #         print("downsample_count = ", downsample_count)
#         downsample_factor = 2**(downsample_count)

#         hop_length //= downsample_factor # Getting new hop_length
#         new_sr = sr / float(downsample_factor) # Getting new sampling rate

#         sr = new_sr

#         return sr, hop_length, downsample_factor
    
    
#     def forward(self,x):
#         x = broadcast_dim(x)
#         if self.earlydownsample==True:
#             x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
#         hop = self.hop_length
#         CQT = self.get_cqt(x, hop, self.padding) #Getting the top octave CQT
        
#         x_down = x # Preparing a new variable for downsampling
#         for i in range(self.n_octaves-1):  
#             hop = hop//2   
#             x_down = downsampling_by_2(x_down, self.lowpass_filter)
#             CQT1 = self.get_cqt(x_down, hop, self.padding)
#             CQT = torch.cat((CQT1, CQT),1) #
#         CQT = CQT[:,-self.n_bins:,:] #Removing unwanted top bins
#         CQT = CQT*2**(self.n_octaves-1) #Normalizing signals with respect to n_fft

#         CQT = CQT*self.downsample_factor/2**(self.n_octaves-1) # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it same mag as 1992
        
#         if self.norm:
#             return CQT/self.n_fft
#         else:
#             return CQT

# class CQT2019(torch.nn.Module):
#     """
#     This alogrithm is using the resampling method proposed in [1]. Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the input audio by a factor of 2 to convoluting it with the small CQT kernel. Everytime the input audio is downsampled, the CQT relative to the downsampled input is equavalent to the next lower octave.
#     The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the code from the 1992 alogrithm [2] 
#     [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
#     [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992).
    
#     early downsampling factor is to downsample the input audio to reduce the CQT kernel size. The result with and without early downsampling are more or less the same except in the very low frequency region where freq < 40Hz
    
#     """
#     def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=True, basis_norm=1, window='hann', pad_mode='reflect', earlydownsample=True):
#         super(CQT2019, self).__init__()
        
#         self.norm = norm # Now norm is used to normalize the final CQT result by dividing n_fft
#         #basis_norm is for normlaizing basis
#         self.hop_length = hop_length
#         self.pad_mode = pad_mode
#         self.n_bins = n_bins   
#         self.earlydownsample = earlydownsample # We will activate eraly downsampling later if possible
        
#         Q = 1/(2**(1/bins_per_octave)-1) # It will be used to calculate filter_cutoff and creating CQT kernels
        
#         # Creating lowpass filter and make it a torch tensor
#         print("Creating low pass filter ...", end='\r')
#         start = time()
#         self.lowpass_filter = torch.tensor( 
#                                             create_lowpass_filter(
#                                             band_center = 0.50, 
#                                             kernelLength=256,
#                                             transitionBandwidth=0.001))
#         self.lowpass_filter = self.lowpass_filter[None,None,:] # Broadcast the tensor to the shape that fits conv1d
#         print("Low pass filter created, time used = {:.4f} seconds".format(time()-start))

#         # Caluate num of filter requires for the kernel
#         # n_octaves determines how many resampling requires for the CQT
#         n_filters = min(bins_per_octave, n_bins)
#         self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
#         print("num_octave = ", self.n_octaves)
        
#         # Calculate the lowest frequency bin for the top octave kernel      
#         self.fmin_t = fmin*2**(self.n_octaves-1)
#         remainder = n_bins % bins_per_octave
# #         print("remainder = ", remainder)
#         if remainder==0:
#             fmax_t = self.fmin_t*2**((bins_per_octave-1)/bins_per_octave) # Calculate the top bin frequency
#         else:
#             fmax_t = self.fmin_t*2**((remainder-1)/bins_per_octave) # Calculate the top bin frequency
#         self.fmin_t = fmax_t/2**(1-1/bins_per_octave) # Adjusting the top minium bins
#         if fmax_t > sr/2:
#             raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins'.format(fmax_t))        
        
#         if self.earlydownsample == True: # Do early downsampling if this argument is True
#             print("Creating early downsampling filter ...", end='\r')
#             start = time()            
#             sr, self.hop_length, self.downsample_factor, self.early_downsample_filter, self.earlydownsample = self.get_early_downsample_params(sr, hop_length, fmax_t, Q, self.n_octaves)
#             print("Early downsampling filter created, time used = {:.4f} seconds".format(time()-start))
#         else:
#             self.downsample_factor=1.
        
#         # Preparing CQT kernels
#         print("Creating CQT kernels ...", end='\r')
#         start = time()
#         basis, self.n_fft, _ = create_cqt_kernels(Q, sr, self.fmin_t, n_filters, bins_per_octave, norm=basis_norm, topbin_check=False)
#         self.basis = basis
#         self.cqt_kernels_real = torch.tensor(basis.real.astype(np.float32)).unsqueeze(1) # These cqt_kernal is already in the frequency domain
#         self.cqt_kernels_imag = torch.tensor(basis.imag.astype(np.float32)).unsqueeze(1)
#         print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
# #         print("Getting cqt kernel done, n_fft = ",self.n_fft)      
        
#         # If center==True, the STFT window will be put in the middle, and paddings at the beginning and ending are required.
#         if self.pad_mode == 'constant':
#             self.padding = nn.ConstantPad1d(self.n_fft//2, 0)
#         elif self.pad_mode == 'reflect':
#             self.padding = nn.ReflectionPad1d(self.n_fft//2)
                
    
#     def get_cqt(self,x,hop_length, padding):
#         """Multiplying the STFT result with the cqt_kernal, check out the 1992 CQT paper [1] for how to multiple the STFT result with the CQT kernel
#         [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992)."""
        
#         # STFT, converting the audio input from time domain to frequency domain
#         try:
#             x = padding(x) # When center == True, we need padding at the beginning and ending
#         except:
#             print("padding with reflection mode might not be the best choice, try using constant padding")
#         CQT_real = conv1d(x, self.cqt_kernels_real, stride=hop_length)
#         CQT_imag = conv1d(x, self.cqt_kernels_imag, stride=hop_length)   
        
#         # Getting CQT Amplitude
#         CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        
#         return CQT

    
#     def get_early_downsample_params(self, sr, hop_length, fmax_t, Q, n_octaves):
#         window_bandwidth = 1.5 # for hann window
#         filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)   
#         sr, hop_length, downsample_factor=self.early_downsample(sr, hop_length, n_octaves, sr//2, filter_cutoff)
#         if downsample_factor != 1:
#             print("Can do early downsample, factor = ", downsample_factor)
#             earlydownsample=True
# #             print("new sr = ", sr)
# #             print("new hop_length = ", hop_length)
#             early_downsample_filter = create_lowpass_filter(band_center=1/downsample_factor, kernelLength=256, transitionBandwidth=0.03)
#             early_downsample_filter = torch.tensor(early_downsample_filter)[None, None, :]
#         else:            
#             print("No early downsampling is required, downsample_factor = ", downsample_factor)
#             early_downsample_filter = None
#             earlydownsample=False
#         return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample    
    
#     # The following two downsampling count functions are obtained from librosa CQT 
#     # They are used to determine the number of pre resamplings if the starting and ending frequency are both in low frequency regions.
#     def early_downsample_count(self, nyquist, filter_cutoff, hop_length, n_octaves):
#         '''Compute the number of early downsampling operations'''

#         downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist /
#                                                        filter_cutoff)) - 1) - 1)
# #         print("downsample_count1 = ", downsample_count1)
#         num_twos = nextpow2(hop_length)
#         downsample_count2 = max(0, num_twos - n_octaves + 1)
# #         print("downsample_count2 = ",downsample_count2)

#         return min(downsample_count1, downsample_count2)

#     def early_downsample(self, sr, hop_length, n_octaves,
#                            nyquist, filter_cutoff):
#         '''Return new sampling rate and hop length after early dowansampling'''
#         downsample_count = self.early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
# #         print("downsample_count = ", downsample_count)
#         downsample_factor = 2**(downsample_count)

#         hop_length //= downsample_factor # Getting new hop_length
#         new_sr = sr / float(downsample_factor) # Getting new sampling rate

#         sr = new_sr

#         return sr, hop_length, downsample_factor
    
    
#     def forward(self,x):
#         x = broadcast_dim(x)
#         if self.earlydownsample==True:
#             x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
#         hop = self.hop_length
#         CQT = self.get_cqt(x, hop, self.padding) #Getting the top octave CQT
        
#         x_down = x # Preparing a new variable for downsampling
#         for i in range(self.n_octaves-1):  
#             hop = hop//2   
#             x_down = downsampling_by_2(x_down, self.lowpass_filter)
#             CQT1 = self.get_cqt(x_down, hop, self.padding)
#             CQT = torch.cat((CQT1, CQT),1) #
#         CQT = CQT[:,-self.n_bins:,:] #Removing unwanted bottom bins
#         CQT = CQT*2**(self.n_octaves-1) #Normalizing signals with respect to n_fft
#         CQT = CQT*self.downsample_factor/2**(self.n_octaves-1) # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it same mag as 1992

#         return CQT
        