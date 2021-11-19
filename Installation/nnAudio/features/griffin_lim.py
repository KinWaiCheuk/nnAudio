import torch.nn as nn
import torch
import numpy as np
from time import time
from ..utils import *


class Griffin_Lim(nn.Module):
    """
    Converting Magnitude spectrograms back to waveforms based on the "fast Griffin-Lim"[1].
    This Griffin Lim is a direct clone from librosa.griffinlim.

    [1] Perraudin, N., Balazs, P., & Søndergaard, P. L. “A fast Griffin-Lim algorithm,”
    IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4), Oct. 2013.

    Parameters
    ----------
    n_fft : int
        The window size. Default value is 2048.

    n_iter=32 : int
        The number of iterations for Griffin-Lim. The default value is ``32``

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.
        Please make sure the value is the same as the forward STFT.

    window : str
        The windowing function for iSTFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.
        Please make sure the value is the same as the forward STFT.

    center : bool
        Putting the iSTFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the iSTFT kernel, if ``True``, the time index is the center of
        the iSTFT kernel. Default value if ``True``.
        Please make sure the value is the same as the forward STFT.

    momentum : float
        The momentum for the update rule. The default value is ``0.99``.

    device : str
        Choose which device to initialize this layer. Default value is 'cpu'

    """

    def __init__(
        self,
        n_fft,
        n_iter=32,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        momentum=0.99,
        device="cpu",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.n_iter = n_iter
        self.center = center
        self.pad_mode = pad_mode
        self.momentum = momentum
        self.device = device
        if win_length == None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        if hop_length == None:
            self.hop_length = n_fft // 4
        else:
            self.hop_length = hop_length

        # Creating window function for stft and istft later
        self.w = torch.tensor(
            get_window(window, int(self.win_length), fftbins=True), device=device
        ).float()

    def forward(self, S):
        """
        Convert a batch of magnitude spectrograms to waveforms.

        Parameters
        ----------
        S : torch tensor
            Spectrogram of the shape ``(batch, n_fft//2+1, timesteps)``
        """

        assert (
            S.dim() == 3
        ), "Please make sure your input is in the shape of (batch, freq_bins, timesteps)"

        # Initializing Random Phase
        rand_phase = torch.randn(*S.shape, device=self.device)
        angles = torch.empty((*S.shape, 2), device=self.device)
        angles[:, :, :, 0] = torch.cos(2 * np.pi * rand_phase)
        angles[:, :, :, 1] = torch.sin(2 * np.pi * rand_phase)

        # Initializing the rebuilt magnitude spectrogram
        rebuilt = torch.zeros(*angles.shape, device=self.device)

        for _ in range(self.n_iter):
            tprev = rebuilt  # Saving previous rebuilt magnitude spec

            # spec2wav conversion
            #             print(f'win_length={self.win_length}\tw={self.w.shape}')
            inverse = torch.istft(
                S.unsqueeze(-1) * angles,
                self.n_fft,
                self.hop_length,
                win_length=self.win_length,
                window=self.w,
                center=self.center,
            )
            # wav2spec conversion
            rebuilt = torch.stft(
                inverse,
                self.n_fft,
                self.hop_length,
                win_length=self.win_length,
                window=self.w,
                pad_mode=self.pad_mode,
            )

            # Phase update rule
            angles[:, :, :] = (
                rebuilt[:, :, :]
                - (self.momentum / (1 + self.momentum)) * tprev[:, :, :]
            )

            # Phase normalization
            angles = angles.div(
                torch.sqrt(angles.pow(2).sum(-1)).unsqueeze(-1) + 1e-16
            )  # normalizing the phase

        # Using the final phase to reconstruct the waveforms
        inverse = torch.istft(
            S.unsqueeze(-1) * angles,
            self.n_fft,
            self.hop_length,
            win_length=self.win_length,
            window=self.w,
            center=self.center,
        )
        return inverse

