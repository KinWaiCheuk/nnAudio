import torch.nn as nn
import torch
import numpy as np
from time import time
from ..utils import *
from .stft import STFT


class MelSpectrogram(nn.Module):
    """This function is to calculate the Melspectrogram of the input signal.
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

    n_mels : int
        The number of Mel filter banks. The filter banks maps the n_fft to mel bins.
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
        The starting frequency for the lowest Mel filter bank.

    fmax : int
        The ending frequency for the highest Mel filter bank.

    norm :
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization, AKA 'slaney' default in librosa).
        Otherwise, leave all the triangles aiming for
        a peak value of 1.0

    trainable_mel : bool
        Determine if the Mel filter banks are trainable or not. If ``True``, the gradients for Mel
        filter banks will also be calculated and the Mel filter banks will be updated during model
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
    >>> spec_layer = Spectrogram.MelSpectrogram()
    >>> specs = spec_layer(x)
    """

    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        win_length=None,
        n_mels=128,
        hop_length=512,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        htk=False,
        fmin=0.0,
        fmax=None,
        norm=1,
        trainable_mel=False,
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
        self.trainable_mel = trainable_mel
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
        mel_basis = get_mel(sr, n_fft, n_mels, fmin, fmax, htk=htk, norm=norm)
        mel_basis = torch.tensor(mel_basis)

        if verbose == True:
            print(
                "STFT filter created, time used = {:.4f} seconds".format(time() - start)
            )
            print(
                "Mel filter created, time used = {:.4f} seconds".format(time() - start)
            )
        else:
            pass

        if trainable_mel:
            # Making everything nn.Parameter, so that this model can support nn.DataParallel
            mel_basis = nn.Parameter(mel_basis, requires_grad=trainable_mel)
            self.register_parameter("mel_basis", mel_basis)
        else:
            self.register_buffer("mel_basis", mel_basis)

        # if trainable_mel==True:
        #     self.mel_basis = nn.Parameter(self.mel_basis)
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

        melspec = torch.matmul(self.mel_basis, spec)
        return melspec

    def extra_repr(self) -> str:
        return "Mel filter banks size = {}, trainable_mel={}".format(
            (*self.mel_basis.shape,), self.trainable_mel, self.trainable_STFT
        )


class MFCC(nn.Module):
    """This function is to calculate the Mel-frequency cepstral coefficients (MFCCs) of the input signal.
    This algorithm first extracts Mel spectrograms from the audio clips,
    then the discrete cosine transform is calcuated to obtain the final MFCCs.
    Therefore, the Mel spectrogram part can be made trainable using
    ``trainable_mel`` and ``trainable_STFT``.
    It only support type-II DCT at the moment. Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio.  It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

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

    def __init__(
        self,
        sr=22050,
        n_mfcc=20,
        norm="ortho",
        verbose=True,
        ref=1.0,
        amin=1e-10,
        top_db=80.0,
        **kwargs
    ):
        super().__init__()
        self.melspec_layer = MelSpectrogram(sr=sr, verbose=verbose, **kwargs)
        self.m_mfcc = n_mfcc

        # attributes that will be used for _power_to_db
        if amin <= 0:
            raise ParameterError("amin must be strictly positive")
        amin = torch.tensor([amin])
        ref = torch.abs(torch.tensor([ref]))
        self.register_buffer("amin", amin)
        self.register_buffer("ref", ref)
        self.top_db = top_db
        self.n_mfcc = n_mfcc

    def _power_to_db(self, S):
        """
        Refer to https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#power_to_db
        for the original implmentation.
        """

        log_spec = 10.0 * torch.log10(torch.max(S, self.amin))
        log_spec -= 10.0 * torch.log10(torch.max(self.amin, self.ref))
        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError("top_db must be non-negative")

            # make the dim same as log_spec so that it can be broadcasted
            batch_wise_max = log_spec.flatten(1).max(1)[0].unsqueeze(1).unsqueeze(1)
            log_spec = torch.max(log_spec, batch_wise_max - self.top_db)

        return log_spec

    def _dct(self, x, norm=None):
        """
        Refer to https://github.com/zh217/torch-dct for the original implmentation.
        """
        x = x.permute(
            0, 2, 1
        )  # make freq the last axis, since dct applies to the frequency axis
        x_shape = x.shape
        N = x_shape[-1]

        v = torch.cat([x[:, :, ::2], x[:, :, 1::2].flip([2])], dim=2)
        Vc = rfft_fn(v, 1, onesided=False)

        # TODO: Can make the W_r and W_i trainable here
        k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, :, 0] * W_r - Vc[:, :, :, 1] * W_i

        if norm == "ortho":
            V[:, :, 0] /= np.sqrt(N) * 2
            V[:, :, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V

        return V.permute(0, 2, 1)  # swapping back the time axis and freq axis

    def forward(self, x):
        """
        Convert a batch of waveforms to MFCC.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """

        x = self.melspec_layer(x)
        x = self._power_to_db(x)
        x = self._dct(x, norm="ortho")[:, : self.m_mfcc, :]
        return x

    def extra_repr(self) -> str:
        return "n_mfcc = {}".format((self.n_mfcc))
