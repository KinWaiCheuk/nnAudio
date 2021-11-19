import torch.nn as nn
import torch
import numpy as np
from time import time
from ..utils import *


class Gammatonegram(nn.Module):
    """
    This function is to calculate the Gammatonegram of the input signal.
    
    Input signal should be in either of the following shapes. 1. ``(len_audio)``, 2. ``(num_audio, len_audio)``, 3. ``(num_audio, 1, len_audio)``. The correct shape will be inferred autommatically if the input follows these 3 shapes. This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.
    n_fft : int
        The window size for the STFT. Default value is 2048
    n_mels : int
        The number of Gammatonegram filter banks. The filter banks maps the n_fft to Gammatone bins. Default value is 64

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
        The starting frequency for the lowest Gammatone filter bank
    fmax : int
        The ending frequency for the highest Gammatone filter bank
    trainable_mel : bool
        Determine if the Gammatone filter banks are trainable or not. If ``True``, the gradients for Mel filter banks will also be caluclated and the Mel filter banks will be updated during model training. Default value is ``False``
    trainable_STFT : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT kernels will also be caluclated and the STFT kernels will be updated during model training. Default value is ``False``

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)``.

    Examples
    --------
    >>> spec_layer = Spectrogram.Gammatonegram()
    >>> specs = spec_layer(x)
    
    """

    def __init__(
        self,
        sr=44100,
        n_fft=2048,
        n_bins=64,
        hop_length=512,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        htk=False,
        fmin=20.0,
        fmax=None,
        norm=1,
        trainable_bins=False,
        trainable_STFT=False,
        verbose=True,
    ):
        super().__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.power = power

        # Create filter windows for stft
        start = time()
        wsin, wcos, self.bins2freq, _, _ = create_fourier_kernels(
            n_fft, freq_bins=None, window=window, freq_scale="no", sr=sr
        )

        wsin = torch.tensor(wsin, dtype=torch.float)
        wcos = torch.tensor(wcos, dtype=torch.float)

        if trainable_STFT:
            wsin = nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter("wsin", wsin)
            self.register_parameter("wcos", wcos)
        else:
            self.register_buffer("wsin", wsin)
            self.register_buffer("wcos", wcos)

            # Creating kenral for Gammatone spectrogram
        start = time()
        gammatone_basis = get_gammatone(sr, n_fft, n_bins, fmin, fmax)
        gammatone_basis = torch.tensor(gammatone_basis)

        if verbose == True:
            print(
                "STFT filter created, time used = {:.4f} seconds".format(time() - start)
            )
            print(
                "Gammatone filter created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )
        else:
            pass
        # Making everything nn.Prarmeter, so that this model can support nn.DataParallel

        if trainable_bins:
            gammatone_basis = nn.Parameter(
                gammatone_basis, requires_grad=trainable_bins
            )
            self.register_parameter("gammatone_basis", gammatone_basis)
        else:
            self.register_buffer("gammatone_basis", gammatone_basis)

        # if trainable_mel==True:
        #     self.mel_basis = nn.Parameter(self.mel_basis)
        # if trainable_STFT==True:
        #     self.wsin = nn.Parameter(self.wsin)
        #     self.wcos = nn.Parameter(self.wcos)

    def forward(self, x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.n_fft // 2, 0)
            elif self.pad_mode == "reflect":
                padding = nn.ReflectionPad1d(self.n_fft // 2)

            x = padding(x)

        spec = (
            torch.sqrt(
                conv1d(x, self.wsin, stride=self.stride).pow(2)
                + conv1d(x, self.wcos, stride=self.stride).pow(2)
            )
            ** self.power
        )  # Doing STFT by using conv1d

        gammatonespec = torch.matmul(self.gammatone_basis, spec)
        return gammatonespec

