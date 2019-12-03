import torch

sz_float = 4    # size of a float
epsilon = 10e-8 # fudge factor for normalization

# ---------------------------Filter design -----------------------------------

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
        Determine the spacing between each frequency bin. When 'linear' or 'log' is used, the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.

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
        Determine the return type. 'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``; 'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``; 'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``. The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'. 
    
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
    def __init__(self, n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050, trainable=False, output_format='Magnitude', device='cuda:0'):
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
        if self.trainable==True:
            self.wsin = torch.nn.Parameter(self.wsin)
            self.wcos = torch.nn.Parameter(self.wcos)
        print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))


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

    def __init__(self, sr=22050, n_fft=2048, n_mels=128, hop_length=512, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False, device='cuda:0'):
        super(MelSpectrogram, self).__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.device = device
        
        # Create filter windows for stft
        start = time()
        wsin, wcos, self.bins2freq, _ = create_fourier_kernels(n_fft, freq_bins=None, window=window, freq_scale='no', sr=sr)
        self.wsin = torch.tensor(wsin, dtype=torch.float, device=self.device)
        self.wcos = torch.tensor(wcos, dtype=torch.float, device=self.device)
        print("STFT filter created, time used = {:.4f} seconds".format(time()-start))

        # Creating kenral for mel spectrogram
        start = time()
        mel_basis = mel(sr, n_fft, n_mels, fmin, fmax, htk=htk, norm=norm)
        self.mel_basis = torch.tensor(mel_basis, device=self.device)
        print("Mel filter created, time used = {:.4f} seconds".format(time()-start))
        
        if trainable_mel==True:
            self.mel_basis = torch.nn.Parameter(self.mel_basis)
        if trainable_STFT==True:
            self.wsin = torch.nn.Parameter(self.wsin)
            self.wcos = torch.nn.Parameter(self.wcos)            
        

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
        Determine the return type. 'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``; 'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``; 'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``. The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'. 

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

    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect', trainable=False, output_format='Magnitude', device='cuda:0'):
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
        
        print("Creating CQT kernels ...", end='\r')
        start = time()
        self.cqt_kernels, self.kernal_width, self.lenghts = create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)
        self.lenghts = self.lenghts.to(device)
        self.cqt_kernels_real = torch.tensor(self.cqt_kernels.real, device=self.device).unsqueeze(1)
        self.cqt_kernels_imag = torch.tensor(self.cqt_kernels.imag, device=self.device).unsqueeze(1)
        if trainable==True:
            self.cqt_kernels_real = torch.nn.Parameter(self.cqt_kernels_real)
            self.cqt_kernels_imag = torch.nn.Parameter(self.cqt_kernels_imag)  
        
        
        print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
        
        # creating kernels for stft
#         self.cqt_kernels_real*=lenghts.unsqueeze(1)/self.kernal_width # Trying to normalize as librosa
#         self.cqt_kernels_imag*=lenghts.unsqueeze(1)/self.kernal_width
               

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


    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84, bins_per_octave=12, norm=True, basis_norm=1, window='hann', pad_mode='reflect', earlydownsample=True, trainable=False, output_format='Magnitude', device='cuda:0'):
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
        print("Low pass filter created, time used = {:.4f} seconds".format(time()-start))

        # Caluate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
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
            print("Creating early downsampling filter ...", end='\r')
            start = time()            
            sr, self.hop_length, self.downsample_factor, self.early_downsample_filter, self.earlydownsample = self.get_early_downsample_params(sr, hop_length, fmax_t, Q, self.n_octaves)
            print("Early downsampling filter created, time used = {:.4f} seconds".format(time()-start))
        else:
            self.downsample_factor=1.
        
        # Preparing CQT kernels
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
        if trainable==True:
            self.cqt_kernels_real = torch.nn.Parameter(self.cqt_kernels_real)
            self.cqt_kernels_imag = torch.nn.Parameter(self.cqt_kernels_imag)          
        
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
    
    def get_early_downsample_params(self, sr, hop_length, fmax_t, Q, n_octaves):
        window_bandwidth = 1.5 # for hann window
        filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)   
        sr, hop_length, downsample_factor=self.early_downsample(sr, hop_length, n_octaves, sr//2, filter_cutoff)
        if downsample_factor != 1:
            print("Can do early downsample, factor = ", downsample_factor)
            earlydownsample=True
#             print("new sr = ", sr)
#             print("new hop_length = ", hop_length)
            early_downsample_filter = create_lowpass_filter(band_center=1/downsample_factor, kernelLength=256, transitionBandwidth=0.03)
            early_downsample_filter = torch.tensor(early_downsample_filter, device=self.device)[None, None, :]
        else:            
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
        
class CQT(CQT1992v2):
    """An abbreviation for CQT1992v2. Please refer to the CQT1992v2 documentation"""
    pass