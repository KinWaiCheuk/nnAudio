import torch.nn as nn
import torch
import numpy as np
from time import time
from ..utils import *
from .stft import STFT


class Gammatonegram(nn.Module):
    """This function is to calculate the Gammatonegram of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio.
        It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    n_fft : int
        The window size for the STFT. Default value is 2048

    win_length : int
        the size of window frame and STFT filter.
        Default: None (treated as equal to n_fft)

    n_bins : int
        The number of Gammatone filter banks. The filter banks maps the n_fft to gammatone bins.
        Default value is 128.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``,
        the time index is the beginning of the STFT kernel, if ``True``, the time index is the
        center of the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    htk : bool
        When ``False`` is used, the Mel scale is quasi-logarithmic. When ``True`` is used, the
        Mel scale is logarithmic. The default value is ``False``.

    fmin : int
        The starting frequency for the lowest Gammatone filter bank.

    fmax : int
        The ending frequency for the highest Gammatone filter bank.

    norm :
        if 1, divide the triangular Gammatone weights by the width of the Gammatone band
        (area normalization, AKA 'slaney' default in librosa).
        Otherwise, leave all the triangles aiming for
        a peak value of 1.0

    trainable_bins : bool
        Determine if the Gammatone filter banks are trainable or not. If ``True``, the gradients for Gammatone
        filter banks will also be calculated and the Gammatone filter banks will be updated during model
        training. Default value is ``False``.

    trainable_STFT : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

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
        sr=22050,
        n_fft=2048,
        win_length=None,
        n_bins=64,
        hop_length=512,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        htk=False, 
        fmin=0.0,
        fmax=None,
        norm=1,
        trainable_bins=False,
        trainable_STFT=False,
        verbose=True,
        **kwargs
    ):

        super().__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.power = power
        self.trainable_bins = trainable_bins
        self.trainable_STFT = trainable_STFT

        # Preparing for the stft layer. No need for center
        self.stft = STFT(
            n_fft=n_fft,
            win_length=win_length,
            freq_bins=None,
            hop_length=hop_length,
            window=window,
            freq_scale="no",
            center=center,
            pad_mode=pad_mode,
            sr=sr,
            trainable=trainable_STFT,
            output_format="Magnitude",
            verbose=verbose,
            **kwargs
        )

        # Create filter windows for stft
        start = time()

        # Creating kernel for mel spectrogram
        start = time()
        gammatone_basis = get_gammatone(sr, n_fft, n_bins, fmin, fmax)
        gammatone_basis = torch.tensor(gammatone_basis)

        if verbose == True:
            print(
                "STFT filter created, time used = {:.4f} seconds".format(time() - start)
            )
            print(
                "Gammatone filter created, time used = {:.4f} seconds".format(time() - start)
            )
        else:
            pass

        if trainable_bins:
            # Making everything nn.Parameter, so that this model can support nn.DataParallel
            gammatone_basis = nn.Parameter(gammatone_basis, requires_grad=trainable_bins)
            self.register_parameter("gammatone_basis", gammatone_basis)
        else:
            self.register_buffer("gammatone_basis", gammatone_basis)

        # if trainable_bins==True:
        #     self.gammatone_basis = nn.Parameter(self.gammatone_basis)
        # if trainable_STFT==True:
        #     self.wsin = nn.Parameter(self.wsin)
        #     self.wcos = nn.Parameter(self.wcos)

    def forward(self, x):
        """
        Convert a batch of waveforms to Mel spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        x = broadcast_dim(x)

        spec = self.stft(x, output_format="Magnitude") ** self.power

        gammatonespec = torch.matmul(self.gammatone_basis, spec)
        return gammatonespec

    def extra_repr(self) -> str:
        return "Gammatone filter banks size = {}, trainable_bins={}".format(
            (*self.gammatone_basis.shape,), self.trainable_bins, self.trainable_STFT
        )