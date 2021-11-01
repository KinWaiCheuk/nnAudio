import numpy as np
import torch
import torch.nn as nn
import time
from ..utils import *
import scipy


class Combined_Frequency_Periodicity(nn.Module):
    """
    Vectorized version of the code in https://github.com/leo-so/VocalMelodyExtPatchCNN/blob/master/MelodyExt.py.
    This feature is described in 'Combining Spectral and Temporal Representations for Multipitch Estimation of Polyphonic Music'
    https://ieeexplore.ieee.org/document/7118691

    Parameters
    ----------
    fr : int
        Frequency resolution. The higher the number, the lower the resolution is.
        Maximum frequency resolution occurs when ``fr=1``. The default value is ``2``

    fs : int
        Sample rate of the input audio clips. The default value is ``16000``

    hop_length : int
        The hop (or stride) size. The default value is ``320``.

    window_size : str
        It is same as ``n_fft`` in other Spectrogram classes. The default value is ``2049``

    fc : int
        Starting frequency. For example, ``fc=80`` means that `Z` starts at 80Hz.
        The default value is ``80``.

    tc : int
        Inverse of ending frequency. For example ``tc=1/8000`` means that `Z` ends at 8000Hz.
        The default value is ``1/8000``.

    g: list
        Coefficients for non-linear activation function. ``len(g)`` should be the number of activation layers.
        Each element in ``g`` is the activation coefficient, for example ``[0.24, 0.6, 1]``.

    device : str
        Choose which device to initialize this layer. Default value is 'cpu'

    Returns
    -------
    Z : torch.tensor
        The Combined Frequency and Period Feature. It is equivalent to ``tfrLF * tfrLQ``

    tfrL0: torch.tensor
        STFT output

    tfrLF: torch.tensor
        Frequency Feature

    tfrLQ: torch.tensor
        Period Feature

    Examples
    --------
    >>> spec_layer = Spectrogram.Combined_Frequency_Periodicity()
    >>> Z, tfrL0, tfrLF, tfrLQ = spec_layer(x)

    """

    def __init__(
        self,
        fr=2,
        fs=16000,
        hop_length=320,
        window_size=2049,
        fc=80,
        tc=1 / 1000,
        g=[0.24, 0.6, 1],
        NumPerOct=48,
    ):
        super().__init__()

        self.window_size = window_size
        self.hop_length = hop_length

        # variables for STFT part
        self.N = int(fs / float(fr))  # Will be used to calculate padding
        self.f = fs * np.linspace(
            0, 0.5, np.round(self.N // 2), endpoint=True
        )  # it won't be used but will be returned
        self.pad_value = self.N - window_size
        # Create window function, always blackmanharris?
        h = scipy.signal.blackmanharris(window_size)  # window function for STFT
        self.register_buffer("h", torch.tensor(h).float())

        # variables for CFP
        self.NumofLayer = np.size(g)
        self.g = g
        self.tc_idx = round(
            fs * tc
        )  # index to filter out top tc_idx and bottom tc_idx bins
        self.fc_idx = round(
            fc / fr
        )  # index to filter out top fc_idx and bottom fc_idx bins
        self.HighFreqIdx = int(round((1 / tc) / fr) + 1)
        self.HighQuefIdx = int(round(fs / fc) + 1)

        # attributes to be returned
        self.f = self.f[: self.HighFreqIdx]
        self.q = np.arange(self.HighQuefIdx) / float(fs)

        # filters for the final step
        freq2logfreq_matrix, quef2logfreq_matrix = self.create_logfreq_matrix(
            self.f, self.q, fr, fc, tc, NumPerOct, fs
        )
        self.register_buffer(
            "freq2logfreq_matrix", torch.tensor(freq2logfreq_matrix).float()
        )
        self.register_buffer(
            "quef2logfreq_matrix", torch.tensor(quef2logfreq_matrix).float()
        )

    def _CFP(self, spec):
        spec = torch.relu(spec).pow(self.g[0])

        if self.NumofLayer >= 2:
            for gc in range(1, self.NumofLayer):
                if np.remainder(gc, 2) == 1:
                    ceps = rfft_fn(spec, 1, onesided=False)[:, :, :, 0] / np.sqrt(
                        self.N
                    )
                    ceps = self.nonlinear_func(ceps, self.g[gc], self.tc_idx)
                else:
                    spec = rfft_fn(ceps, 1, onesided=False)[:, :, :, 0] / np.sqrt(
                        self.N
                    )
                    spec = self.nonlinear_func(spec, self.g[gc], self.fc_idx)

        return spec, ceps

    def forward(self, x):
        tfr0 = torch.stft(
            x,
            self.N,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=self.h,
            onesided=False,
            pad_mode="constant",
        )
        tfr0 = torch.sqrt(tfr0.pow(2).sum(-1)) / torch.norm(
            self.h
        )  # calcuate magnitude
        tfr0 = tfr0.transpose(1, 2)[
            :, 1:-1
        ]  # transpose F and T axis and discard first and last frames
        # The transpose is necessary for rfft later
        # (batch, timesteps, n_fft)
        tfr, ceps = self._CFP(tfr0)

        #         return tfr0
        # removing duplicate bins
        tfr0 = tfr0[:, :, : int(round(self.N / 2))]
        tfr = tfr[:, :, : int(round(self.N / 2))]
        ceps = ceps[:, :, : int(round(self.N / 2))]

        # Crop up to the highest frequency
        tfr0 = tfr0[:, :, : self.HighFreqIdx]
        tfr = tfr[:, :, : self.HighFreqIdx]
        ceps = ceps[:, :, : self.HighQuefIdx]
        tfrL0 = torch.matmul(self.freq2logfreq_matrix, tfr0.transpose(1, 2))
        tfrLF = torch.matmul(self.freq2logfreq_matrix, tfr.transpose(1, 2))
        tfrLQ = torch.matmul(self.quef2logfreq_matrix, ceps.transpose(1, 2))
        Z = tfrLF * tfrLQ

        # Only need to calculate this once
        self.t = np.arange(
            self.hop_length,
            np.ceil(len(x) / float(self.hop_length)) * self.hop_length,
            self.hop_length,
        )  # it won't be used but will be returned

        return Z, tfrL0, tfrLF, tfrLQ

    def nonlinear_func(self, X, g, cutoff):
        cutoff = int(cutoff)
        if g != 0:
            X = torch.relu(X)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
            X = X.pow(g)
        else:  # when g=0, it converges to log
            X = torch.log(X.relu() + epsilon)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
        return X

    def create_logfreq_matrix(self, f, q, fr, fc, tc, NumPerOct, fs):
        StartFreq = fc
        StopFreq = 1 / tc
        Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
        central_freq = []  # A list holding the frequencies in log scale

        for i in range(0, Nest):
            CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
            if CenFreq < StopFreq:
                central_freq.append(CenFreq)
            else:
                break

        Nest = len(central_freq)
        freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)

        # Calculating the freq_band_transformation
        for i in range(1, Nest - 1):
            l = int(round(central_freq[i - 1] / fr))
            r = int(round(central_freq[i + 1] / fr) + 1)
            # rounding1
            if l >= r - 1:
                freq_band_transformation[i, l] = 1
            else:
                for j in range(l, r):
                    if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                        freq_band_transformation[i, j] = (
                            f[j] - central_freq[i - 1]
                        ) / (central_freq[i] - central_freq[i - 1])
                    elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                        freq_band_transformation[i, j] = (
                            central_freq[i + 1] - f[j]
                        ) / (central_freq[i + 1] - central_freq[i])

        # Calculating the quef_band_transformation
        f = 1 / q  # divide by 0, do I need to fix this?
        quef_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
        for i in range(1, Nest - 1):
            for j in range(
                int(round(fs / central_freq[i + 1])),
                int(round(fs / central_freq[i - 1]) + 1),
            ):
                if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                    quef_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (
                        central_freq[i] - central_freq[i - 1]
                    )
                elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                    quef_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (
                        central_freq[i + 1] - central_freq[i]
                    )

        return freq_band_transformation, quef_band_transformation


class CFP(nn.Module):
    """
    This is the modified version of :func:`~nnAudio.Spectrogram.Combined_Frequency_Periodicity`. This version different from the original version by returnning only ``Z`` and the number of timesteps fits with other classes.

    Parameters
    ----------
    fr : int
        Frequency resolution. The higher the number, the lower the resolution is.
        Maximum frequency resolution occurs when ``fr=1``. The default value is ``2``

    fs : int
        Sample rate of the input audio clips. The default value is ``16000``

    hop_length : int
        The hop (or stride) size. The default value is ``320``.

    window_size : str
        It is same as ``n_fft`` in other Spectrogram classes. The default value is ``2049``

    fc : int
        Starting frequency. For example, ``fc=80`` means that `Z` starts at 80Hz.
        The default value is ``80``.

    tc : int
        Inverse of ending frequency. For example ``tc=1/8000`` means that `Z` ends at 8000Hz.
        The default value is ``1/8000``.

    g: list
        Coefficients for non-linear activation function. ``len(g)`` should be the number of activation layers.
        Each element in ``g`` is the activation coefficient, for example ``[0.24, 0.6, 1]``.

    device : str
        Choose which device to initialize this layer. Default value is 'cpu'

    Returns
    -------
    Z : torch.tensor
        The Combined Frequency and Period Feature. It is equivalent to ``tfrLF * tfrLQ``

    tfrL0: torch.tensor
        STFT output

    tfrLF: torch.tensor
        Frequency Feature

    tfrLQ: torch.tensor
        Period Feature

    Examples
    --------
    >>> spec_layer = Spectrogram.Combined_Frequency_Periodicity()
    >>> Z, tfrL0, tfrLF, tfrLQ = spec_layer(x)

    """

    def __init__(
        self,
        fr=2,
        fs=16000,
        hop_length=320,
        window_size=2049,
        fc=80,
        tc=1 / 1000,
        g=[0.24, 0.6, 1],
        NumPerOct=48,
    ):
        super().__init__()

        self.window_size = window_size
        self.hop_length = hop_length

        # variables for STFT part
        self.N = int(fs / float(fr))  # Will be used to calculate padding
        self.f = fs * np.linspace(
            0, 0.5, np.round(self.N // 2), endpoint=True
        )  # it won't be used but will be returned
        self.pad_value = self.N - window_size
        # Create window function, always blackmanharris?
        h = scipy.signal.blackmanharris(window_size)  # window function for STFT
        self.register_buffer("h", torch.tensor(h).float())

        # variables for CFP
        self.NumofLayer = np.size(g)
        self.g = g
        self.tc_idx = round(
            fs * tc
        )  # index to filter out top tc_idx and bottom tc_idx bins
        self.fc_idx = round(
            fc / fr
        )  # index to filter out top fc_idx and bottom fc_idx bins
        self.HighFreqIdx = int(round((1 / tc) / fr) + 1)
        self.HighQuefIdx = int(round(fs / fc) + 1)

        # attributes to be returned
        self.f = self.f[: self.HighFreqIdx]
        self.q = np.arange(self.HighQuefIdx) / float(fs)

        # filters for the final step
        freq2logfreq_matrix, quef2logfreq_matrix = self.create_logfreq_matrix(
            self.f, self.q, fr, fc, tc, NumPerOct, fs
        )
        self.register_buffer(
            "freq2logfreq_matrix", torch.tensor(freq2logfreq_matrix).float()
        )
        self.register_buffer(
            "quef2logfreq_matrix", torch.tensor(quef2logfreq_matrix).float()
        )

    def _CFP(self, spec):
        spec = torch.relu(spec).pow(self.g[0])

        if self.NumofLayer >= 2:
            for gc in range(1, self.NumofLayer):
                if np.remainder(gc, 2) == 1:
                    ceps = rfft_fn(spec, 1, onesided=False)[:, :, :, 0] / np.sqrt(
                        self.N
                    )
                    ceps = self.nonlinear_func(ceps, self.g[gc], self.tc_idx)
                else:
                    spec = rfft_fn(ceps, 1, onesided=False)[:, :, :, 0] / np.sqrt(
                        self.N
                    )
                    spec = self.nonlinear_func(spec, self.g[gc], self.fc_idx)

        return spec, ceps

    def forward(self, x):
        tfr0 = torch.stft(
            x,
            self.N,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=self.h,
            onesided=False,
            pad_mode="constant",
        )
        tfr0 = torch.sqrt(tfr0.pow(2).sum(-1)) / torch.norm(
            self.h
        )  # calcuate magnitude
        tfr0 = tfr0.transpose(
            1, 2
        )  # transpose F and T axis and discard first and last frames
        # The transpose is necessary for rfft later
        # (batch, timesteps, n_fft)
        tfr, ceps = self._CFP(tfr0)

        #         return tfr0
        # removing duplicate bins
        tfr0 = tfr0[:, :, : int(round(self.N / 2))]
        tfr = tfr[:, :, : int(round(self.N / 2))]
        ceps = ceps[:, :, : int(round(self.N / 2))]

        # Crop up to the highest frequency
        tfr0 = tfr0[:, :, : self.HighFreqIdx]
        tfr = tfr[:, :, : self.HighFreqIdx]
        ceps = ceps[:, :, : self.HighQuefIdx]
        tfrL0 = torch.matmul(self.freq2logfreq_matrix, tfr0.transpose(1, 2))
        tfrLF = torch.matmul(self.freq2logfreq_matrix, tfr.transpose(1, 2))
        tfrLQ = torch.matmul(self.quef2logfreq_matrix, ceps.transpose(1, 2))
        Z = tfrLF * tfrLQ

        # Only need to calculate this once
        self.t = np.arange(
            self.hop_length,
            np.ceil(len(x) / float(self.hop_length)) * self.hop_length,
            self.hop_length,
        )  # it won't be used but will be returned

        return Z  # , tfrL0, tfrLF, tfrLQ

    def nonlinear_func(self, X, g, cutoff):
        cutoff = int(cutoff)
        if g != 0:
            X = torch.relu(X)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
            X = X.pow(g)
        else:  # when g=0, it converges to log
            X = torch.log(X.relu() + epsilon)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
        return X

    def create_logfreq_matrix(self, f, q, fr, fc, tc, NumPerOct, fs):
        StartFreq = fc
        StopFreq = 1 / tc
        Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
        central_freq = []  # A list holding the frequencies in log scale

        for i in range(0, Nest):
            CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
            if CenFreq < StopFreq:
                central_freq.append(CenFreq)
            else:
                break

        Nest = len(central_freq)
        freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)

        # Calculating the freq_band_transformation
        for i in range(1, Nest - 1):
            l = int(round(central_freq[i - 1] / fr))
            r = int(round(central_freq[i + 1] / fr) + 1)
            # rounding1
            if l >= r - 1:
                freq_band_transformation[i, l] = 1
            else:
                for j in range(l, r):
                    if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                        freq_band_transformation[i, j] = (
                            f[j] - central_freq[i - 1]
                        ) / (central_freq[i] - central_freq[i - 1])
                    elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                        freq_band_transformation[i, j] = (
                            central_freq[i + 1] - f[j]
                        ) / (central_freq[i + 1] - central_freq[i])

        # Calculating the quef_band_transformation
        f = 1 / q  # divide by 0, do I need to fix this?
        quef_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
        for i in range(1, Nest - 1):
            for j in range(
                int(round(fs / central_freq[i + 1])),
                int(round(fs / central_freq[i - 1]) + 1),
            ):
                if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                    quef_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (
                        central_freq[i] - central_freq[i - 1]
                    )
                elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                    quef_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (
                        central_freq[i + 1] - central_freq[i]
                    )

        return freq_band_transformation, quef_band_transformation
