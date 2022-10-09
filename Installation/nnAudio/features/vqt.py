import torch
import torch.nn as nn
import numpy as np
from time import time
from ..librosa_functions import * 
from ..utils import * 


class VQT(torch.nn.Module):
    def __init__(
        self, 
        sr=22050, 
        hop_length=512, 
        fmin=32.70, 
        fmax=None, 
        n_bins=84, 
        filter_scale=1,
        bins_per_octave=12, 
        norm=True, 
        basis_norm=1, 
        gamma=0, 
        window='hann', 
        pad_mode='reflect',
        earlydownsample=True, 
        trainable=False, 
        output_format='Magnitude', 
        verbose=True
    ):

        super().__init__()

        self.norm = norm
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.earlydownsample = earlydownsample
        self.trainable = trainable
        self.output_format = output_format
        self.filter_scale = filter_scale
        self.bins_per_octave = bins_per_octave
        self.sr = sr
        self.gamma = gamma
        self.basis_norm = basis_norm

        # It will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(filter_scale)/(2**(1/bins_per_octave)-1)

        # Creating lowpass filter and make it a torch tensor
        if verbose==True:
            print("Creating low pass filter ...", end='\r')
        start = time()
        lowpass_filter = torch.tensor(create_lowpass_filter(
                                                            band_center = 0.50,
                                                            kernelLength=256,
                                                            transitionBandwidth=0.001)
                                                            )

        self.register_buffer('lowpass_filter', lowpass_filter[None,None,:])
        if verbose == True:
            print("Low pass filter created, time used = {:.4f} seconds".format(time()-start))

        n_filters = min(bins_per_octave, n_bins)
        self.n_filters = n_filters
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        if verbose == True:
            print("num_octave = ", self.n_octaves)

        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave

        if remainder==0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t*2**((bins_per_octave-1)/bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t*2**((remainder-1)/bins_per_octave)

        # Adjusting the top minimum bins
        self.fmin_t = fmax_t / 2 ** (1 - 1 / bins_per_octave) 
        if fmax_t > sr/2:
            raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, \
                            please reduce the n_bins'.format(fmax_t))

        if self.earlydownsample == True: # Do early downsampling if this argument is True
            if verbose == True:
                print("Creating early downsampling filter ...", end='\r')
            start = time()
            sr, self.hop_length, self.downsample_factor, early_downsample_filter, \
                self.earlydownsample = get_early_downsample_params(sr,
                                                                   hop_length,
                                                                   fmax_t,
                                                                   Q,
                                                                   self.n_octaves,
                                                                   verbose)
            self.register_buffer('early_downsample_filter', early_downsample_filter)
            
            if verbose==True:
                print("Early downsampling filter created, \
                        time used = {:.4f} seconds".format(time()-start))
        else:
            self.downsample_factor = 1.
        
        # For normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs
        lenghts = np.ceil(Q * sr / (freqs + gamma / alpha))
        
        # get max window length depending on gamma value
        max_len = int(max(lenghts))
        self.n_fft = int(2 ** (np.ceil(np.log2(max_len))))

        lenghts = torch.tensor(lenghts).float()
        self.register_buffer('lenghts', lenghts)
        
        
        my_sr = self.sr
        for i in range(self.n_octaves):
            if i > 0:
                my_sr /= 2          
            
            Q = float(self.filter_scale)/(2**(1/self.bins_per_octave)-1)

            basis, self.n_fft, lengths, _ = create_cqt_kernels(Q,
                                                               my_sr, 
                                                               self.fmin_t * 2 ** -i,
                                                               self.n_filters,
                                                               self.bins_per_octave,
                                                               norm=self.basis_norm,
                                                               topbin_check=False,
                                                               gamma=self.gamma)
            
            cqt_kernels_real = torch.tensor(basis.real.astype(np.float32)).unsqueeze(1)
            cqt_kernels_imag = torch.tensor(basis.imag.astype(np.float32)).unsqueeze(1)
            
            self.register_buffer("cqt_kernels_real_{}".format(i), cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag_{}".format(i), cqt_kernels_imag)            
            

    def forward(self, x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to VQT spectrograms.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """              
        output_format = output_format or self.output_format
        
        x = broadcast_dim(x)
        if self.earlydownsample==True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        vqt = []

        x_down = x  # Preparing a new variable for downsampling
        my_sr = self.sr

        for i in range(self.n_octaves):
            if i > 0:
                x_down = downsampling_by_2(x_down, self.lowpass_filter)
                hop //= 2

            else:
                x_down = x

            if self.pad_mode == 'constant':
                my_padding = nn.ConstantPad1d(getattr(self,
                                                      'cqt_kernels_real_{}'.format(i)).shape[-1] // 2, 0)
            elif self.pad_mode == 'reflect':
                my_padding= nn.ReflectionPad1d(getattr(self,
                                                       'cqt_kernels_real_{}'.format(i)).shape[-1] // 2)

            cur_vqt = get_cqt_complex(x_down,
                                      getattr(self, 'cqt_kernels_real_{}'.format(i)),
                                      getattr(self, 'cqt_kernels_imag_{}'.format(i)),
                                      hop,
                                      my_padding)
            vqt.insert(0, cur_vqt)

        vqt = torch.cat(vqt, dim=1)
        vqt = vqt[:,-self.n_bins:,:]    # Removing unwanted bottom bins
        vqt = vqt * self.downsample_factor

        # Normalize again to get same result as librosa
        if normalization_type == 'librosa':
            vqt = vqt * torch.sqrt(self.lenghts.view(-1,1,1))
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            vqt *= 2
        else:
            raise ValueError("The normalization_type %r is not part of our current options." % normalization_type)

        if output_format=='Magnitude':
            if self.trainable==False:
                # Getting CQT Amplitude
                return torch.sqrt(vqt.pow(2).sum(-1))
            else:
                return torch.sqrt(vqt.pow(2).sum(-1) + 1e-8)

        elif output_format=='Complex':
            return vqt

        elif output_format=='Phase':
            phase_real = torch.cos(torch.atan2(vqt[:,:,:,1], vqt[:,:,:,0]))
            phase_imag = torch.sin(torch.atan2(vqt[:,:,:,1], vqt[:,:,:,0]))
            return torch.stack((phase_real,phase_imag), -1)