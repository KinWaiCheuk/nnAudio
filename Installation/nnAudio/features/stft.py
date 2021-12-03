from torch.nn.functional import conv1d, conv2d, fold
import torch.nn as nn
import torch
import numpy as np
from time import time
from ..utils import *
from scipy.signal import get_window


class STFTBase(nn.Module):
    """
    STFT and iSTFT share the same `inverse_stft` function
    """

    def inverse_stft(
        self, X, kernel_cos, kernel_sin, onesided=True, length=None, refresh_win=True
    ):
        # If the input spectrogram contains only half of the n_fft
        # Use extend_fbins function to get back another half
        if onesided:
            X = extend_fbins(X)  # extend freq
        X_real, X_imag = X[:, :, :, 0], X[:, :, :, 1]

        # broadcast dimensions to support 2D convolution
        X_real_bc = X_real.unsqueeze(1)
        X_imag_bc = X_imag.unsqueeze(1)
        a1 = conv2d(X_real_bc, kernel_cos, stride=(1, 1))
        b2 = conv2d(X_imag_bc, kernel_sin, stride=(1, 1))
        # compute real and imag part. signal lies in the real part
        real = a1 - b2
        real = real.squeeze(-2) * self.window_mask

        # Normalize the amplitude with n_fft
        real /= self.n_fft

        # Overlap and Add algorithm to connect all the frames
        real = overlap_add(real, self.stride)

        # Prepare the window sumsqure for division
        # Only need to create this window once to save time
        # Unless the input spectrograms have different time steps
        if hasattr(self, "w_sum") == False or refresh_win == True:
            self.w_sum = torch_window_sumsquare(
                self.window_mask.flatten(), X.shape[2], self.stride, self.n_fft
            ).flatten()
            self.nonzero_indices = self.w_sum > 1e-10
        else:
            pass
        real[:, self.nonzero_indices] = real[:, self.nonzero_indices].div(
            self.w_sum[self.nonzero_indices]
        )
        # Remove padding
        if length is None:
            if self.center:
                real = real[:, self.pad_amount : -self.pad_amount]

        else:
            if self.center:
                real = real[:, self.pad_amount : self.pad_amount + length]
            else:
                real = real[:, :length]

        return real

    ### --------------------------- Spectrogram Classes ---------------------------###


class STFT(STFTBase):
    """This function is to calculate the short-time Fourier transform (STFT) of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    Parameters
    ----------
    n_fft : int
        Size of Fourier transform. Default value is 2048.

    win_length : int
        the size of window frame and STFT filter.
        Default: None (treated as equal to n_fft)

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins.

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    freq_scale : 'linear', 'log', 'log2' or 'no'
        Determine the spacing between each frequency bin. When `linear`, 'log' or `log2` is used,
        the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will
        start at 0Hz and end at Nyquist frequency with linear spacing.

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the STFT kernel, if ``True``, the time index is the center of
        the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    iSTFT : bool
        To activate the iSTFT module or not. By default, it is False to save GPU memory.
        Note: The iSTFT kernel is not trainable. If you want
        a trainable iSTFT, use the iSTFT module.

    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument
        does nothing.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument
        does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    trainable : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``

    output_format : str
        Control the spectrogram output type, either ``Magnitude``, ``Complex``, or ``Phase``.
        The output_format can also be changed during the ``forward`` method.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.
        ``shape = (num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
        ``shape = (num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.STFT()
    >>> specs = spec_layer(x)
    """

    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        freq_bins=None,
        hop_length=None,
        window="hann",
        freq_scale="no",
        center=True,
        pad_mode="reflect",
        iSTFT=False,
        fmin=50,
        fmax=6000,
        sr=22050,
        trainable=False,
        output_format="Complex",
        verbose=True,
    ):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)

        self.output_format = output_format
        self.trainable = trainable
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length
        self.iSTFT = iSTFT
        self.trainable = trainable
        start = time()

        # Create filter windows for stft
        (
            kernel_sin,
            kernel_cos,
            self.bins2freq,
            self.bin_list,
            window_mask,
        ) = create_fourier_kernels(
            n_fft,
            win_length=win_length,
            freq_bins=freq_bins,
            window=window,
            freq_scale=freq_scale,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            verbose=verbose,
        )

        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)

        # In this way, the inverse kernel and the forward kernel do not share the same memory...
        kernel_sin_inv = torch.cat((kernel_sin, -kernel_sin[1:-1].flip(0)), 0)
        kernel_cos_inv = torch.cat((kernel_cos, kernel_cos[1:-1].flip(0)), 0)

        if iSTFT:
            self.register_buffer("kernel_sin_inv", kernel_sin_inv.unsqueeze(-1))
            self.register_buffer("kernel_cos_inv", kernel_cos_inv.unsqueeze(-1))

        # Making all these variables nn.Parameter, so that the model can be used with nn.Parallel
        #         self.kernel_sin = nn.Parameter(self.kernel_sin, requires_grad=self.trainable)
        #         self.kernel_cos = nn.Parameter(self.kernel_cos, requires_grad=self.trainable)

        # Applying window functions to the Fourier kernels
        window_mask = torch.tensor(window_mask)
        wsin = kernel_sin * window_mask
        wcos = kernel_cos * window_mask

        if self.trainable == False:
            self.register_buffer("wsin", wsin)
            self.register_buffer("wcos", wcos)

        if self.trainable == True:
            wsin = nn.Parameter(wsin, requires_grad=self.trainable)
            wcos = nn.Parameter(wcos, requires_grad=self.trainable)
            self.register_parameter("wsin", wsin)
            self.register_parameter("wcos", wcos)

            # Prepare the shape of window mask so that it can be used later in inverse
        self.register_buffer("window_mask", window_mask.unsqueeze(0).unsqueeze(-1))

        if verbose == True:
            print(
                "STFT kernels created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )
        else:
            pass

    def forward(self, x, output_format=None):
        """
        Convert a batch of waveforms to spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape

        output_format : str
            Control the type of spectrogram to be return. Can be either ``Magnitude`` or ``Complex`` or ``Phase``.
            Default value is ``Complex``.

        """
        output_format = output_format or self.output_format
        self.num_samples = x.shape[-1]

        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.pad_amount, 0)

            elif self.pad_mode == "reflect":
                if self.num_samples < self.pad_amount:
                    raise AssertionError(
                        "Signal length shorter than reflect padding length (n_fft // 2)."
                    )
                padding = nn.ReflectionPad1d(self.pad_amount)

            x = padding(x)
        spec_imag = conv1d(x, self.wsin, stride=self.stride)
        spec_real = conv1d(
            x, self.wcos, stride=self.stride
        )  # Doing STFT by using conv1d

        # remove redundant parts
        spec_real = spec_real[:, : self.freq_bins, :]
        spec_imag = spec_imag[:, : self.freq_bins, :]

        if output_format == "Magnitude":
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable == True:
                return torch.sqrt(
                    spec + 1e-8
                )  # prevent Nan gradient when sqrt(0) due to output=0
            else:
                return torch.sqrt(spec)

        elif output_format == "Complex":
            return torch.stack(
                (spec_real, -spec_imag), -1
            )  # Remember the minus sign for imaginary part

        elif output_format == "Phase":
            return torch.atan2(
                -spec_imag + 0.0, spec_real
            )  # +0.0 removes -0.0 elements, which leads to error in calculating phase

    def inverse(self, X, onesided=True, length=None, refresh_win=True):
        """
        This function is same as the :func:`~nnAudio.Spectrogram.iSTFT` class,
        which is to convert spectrograms back to waveforms.
        It only works for the complex value spectrograms. If you have the magnitude spectrograms,
        please use :func:`~nnAudio.Spectrogram.Griffin_Lim`.

        Parameters
        ----------
        onesided : bool
            If your spectrograms only have ``n_fft//2+1`` frequency bins, please use ``onesided=True``,
            else use ``onesided=False``

        length : int
            To make sure the inverse STFT has the same output length of the original waveform, please
            set `length` as your intended waveform length. By default, ``length=None``,
            which will remove ``n_fft//2`` samples from the start and the end of the output.

        refresh_win : bool
            Recalculating the window sum square. If you have an input with fixed number of timesteps,
            you can increase the speed by setting ``refresh_win=False``. Else please keep ``refresh_win=True``


        """
        if (hasattr(self, "kernel_sin_inv") != True) or (
            hasattr(self, "kernel_cos_inv") != True
        ):
            raise NameError(
                "Please activate the iSTFT module by setting `iSTFT=True` if you want to use `inverse`"
            )

        assert X.dim() == 4, (
            "Inverse iSTFT only works for complex number,"
            "make sure our tensor is in the shape of (batch, freq_bins, timesteps, 2)."
            "\nIf you have a magnitude spectrogram, please consider using Griffin-Lim."
        )
        return self.inverse_stft(
            X, self.kernel_cos_inv, self.kernel_sin_inv, onesided, length, refresh_win
        )

    def extra_repr(self) -> str:
        return "n_fft={}, Fourier Kernel size={}, iSTFT={}, trainable={}".format(
            self.n_fft, (*self.wsin.shape,), self.iSTFT, self.trainable
        )


class iSTFT(STFTBase):
    """This class is to convert spectrograms back to waveforms. It only works for the complex value spectrograms.
    If you have the magnitude spectrograms, please use :func:`~nnAudio.Spectrogram.Griffin_Lim`.
    The parameters (e.g. n_fft, window) need to be the same as the STFT in order to obtain the correct inverse.
    If trainability is not required, it is recommended to use the ``inverse`` method under the ``STFT`` class
    to save GPU/RAM memory.

    When ``trainable=True`` and ``freq_scale!='no'``, there is no guarantee that the inverse is perfect, please
    use with extra care.

    Parameters
    ----------
    n_fft : int
        The window size. Default value is 2048.

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins
        Please make sure the value is the same as the forward STFT.

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.
        Please make sure the value is the same as the forward STFT.

    window : str
        The windowing function for iSTFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.
        Please make sure the value is the same as the forward STFT.

    freq_scale : 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin. When `linear` or `log` is used,
        the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will
        start at 0Hz and end at Nyquist frequency with linear spacing.
        Please make sure the value is the same as the forward STFT.

    center : bool
        Putting the iSTFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the iSTFT kernel, if ``True``, the time index is the center of
        the iSTFT kernel. Default value if ``True``.
        Please make sure the value is the same as the forward STFT.

    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument
        does nothing. Please make sure the value is the same as the forward STFT.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument
        does nothing. Please make sure the value is the same as the forward STFT.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    trainable_kernels : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``.

    trainable_window : bool
        Determine if the window function is trainable or not.
        Default value is ``False``.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
        It returns a batch of waveforms.

    Examples
    --------
    >>> spec_layer = Spectrogram.iSTFT()
    >>> specs = spec_layer(x)
    """

    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        freq_bins=None,
        hop_length=None,
        window="hann",
        freq_scale="no",
        center=True,
        fmin=50,
        fmax=6000,
        sr=22050,
        trainable_kernels=False,
        trainable_window=False,
        verbose=True,
        refresh_win=True,
    ):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)

        self.n_fft = n_fft
        self.win_length = win_length
        self.stride = hop_length
        self.center = center

        self.pad_amount = self.n_fft // 2
        self.refresh_win = refresh_win

        start = time()

        # Create the window function and prepare the shape for batch-wise-time-wise multiplication

        # Create filter windows for inverse
        kernel_sin, kernel_cos, _, _, window_mask = create_fourier_kernels(
            n_fft,
            win_length=win_length,
            freq_bins=n_fft,
            window=window,
            freq_scale=freq_scale,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            verbose=False,
        )
        window_mask = get_window(window, int(win_length), fftbins=True)

        # For inverse, the Fourier kernels do not need to be windowed
        window_mask = torch.tensor(window_mask).unsqueeze(0).unsqueeze(-1)

        # kernel_sin and kernel_cos have the shape (freq_bins, 1, n_fft, 1) to support 2D Conv
        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float).unsqueeze(-1)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float).unsqueeze(-1)

        # Decide if the Fourier kernels are trainable
        if trainable_kernels:
            # Making all these variables trainable
            kernel_sin = nn.Parameter(kernel_sin, requires_grad=trainable_kernels)
            kernel_cos = nn.Parameter(kernel_cos, requires_grad=trainable_kernels)
            self.register_parameter("kernel_sin", kernel_sin)
            self.register_parameter("kernel_cos", kernel_cos)

        else:
            self.register_buffer("kernel_sin", kernel_sin)
            self.register_buffer("kernel_cos", kernel_cos)

        # Decide if the window function is trainable
        if trainable_window:
            window_mask = nn.Parameter(window_mask, requires_grad=trainable_window)
            self.register_parameter("window_mask", window_mask)
        else:
            self.register_buffer("window_mask", window_mask)

        if verbose == True:
            print(
                "iSTFT kernels created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )
        else:
            pass

    def forward(self, X, onesided=False, length=None, refresh_win=None):
        """
        If your spectrograms only have ``n_fft//2+1`` frequency bins, please use ``onesided=True``,
        else use ``onesided=False``
        To make sure the inverse STFT has the same output length of the original waveform, please
        set `length` as your intended waveform length. By default, ``length=None``,
        which will remove ``n_fft//2`` samples from the start and the end of the output.
        If your input spectrograms X are of the same length, please use ``refresh_win=None`` to increase
        computational speed.
        """
        if refresh_win == None:
            refresh_win = self.refresh_win

        assert X.dim() == 4, (
            "Inverse iSTFT only works for complex number,"
            "make sure our tensor is in the shape of (batch, freq_bins, timesteps, 2)"
        )

        return self.inverse_stft(
            X, self.kernel_cos, self.kernel_sin, onesided, length, refresh_win
        )
