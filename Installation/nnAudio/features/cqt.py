import torch.nn as nn
import torch
import numpy as np
from time import time
from ..utils import *
from ..utils import *


class CQT1992(nn.Module):
    """
    This alogrithm uses the method proposed in [1], which would run extremely slow if low frequencies (below 220Hz)
    are included in the frequency bins.
    Please refer to :func:`~nnAudio.Spectrogram.CQT1992v2` for a more
    computational and memory efficient version.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.



    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    trainable_STFT : bool
        Determine if the time to frequency domain transformation kernel for the input audio is trainable or not.
        Default is ``False``

    trainable_CQT : bool
        Determine if the frequency domain CQT kernel is trainable or not.
        Default is ``False``

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``.

     output_format : str
        Determine the return type.
        ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``;
        ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``;
        ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT1992v2()
    >>> specs = spec_layer(x)
    """

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=220,
        fmax=None,
        n_bins=84,
        trainable_STFT=False,
        trainable_CQT=False,
        bins_per_octave=12,
        filter_scale=1,
        output_format="Magnitude",
        norm=1,
        window="hann",
        center=True,
        pad_mode="reflect",
    ):

        super().__init__()

        # norm arg is not functioning
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        self.output_format = output_format

        # creating kernels for CQT
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        print("Creating CQT kernels ...", end="\r")
        start = time()
        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(
            Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax
        )

        self.register_buffer("lenghts", lenghts)
        self.frequencies = freqs

        cqt_kernels = fft(cqt_kernels)[:, : self.kernel_width // 2 + 1]
        print("CQT kernels created, time used = {:.4f} seconds".format(time() - start))

        # creating kernels for stft
        # self.cqt_kernels_real*=lenghts.unsqueeze(1)/self.kernel_width # Trying to normalize as librosa
        # self.cqt_kernels_imag*=lenghts.unsqueeze(1)/self.kernel_width

        print("Creating STFT kernels ...", end="\r")
        start = time()
        kernel_sin, kernel_cos, self.bins2freq, _, window = create_fourier_kernels(
            self.kernel_width, window="ones", freq_scale="no"
        )

        # Converting kernels from numpy arrays to torch tensors
        wsin = torch.tensor(kernel_sin * window)
        wcos = torch.tensor(kernel_cos * window)

        cqt_kernels_real = torch.tensor(cqt_kernels.real)
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag)

        if trainable_STFT:
            wsin = nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter("wsin", wsin)
            self.register_parameter("wcos", wcos)
        else:
            self.register_buffer("wsin", wsin)
            self.register_buffer("wcos", wcos)

        if trainable_CQT:
            cqt_kernels_real = nn.Parameter(
                cqt_kernels_real, requires_grad=trainable_CQT
            )
            cqt_kernels_imag = nn.Parameter(
                cqt_kernels_imag, requires_grad=trainable_CQT
            )
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

        print("STFT kernels created, time used = {:.4f} seconds".format(time() - start))

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

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
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == "reflect":
                padding = nn.ReflectionPad1d(self.kernel_width // 2)

            x = padding(x)

            # STFT
        fourier_real = conv1d(x, self.wcos, stride=self.hop_length)
        fourier_imag = conv1d(x, self.wsin, stride=self.hop_length)

        # CQT
        CQT_real, CQT_imag = complex_mul(
            (self.cqt_kernels_real, self.cqt_kernels_imag), (fourier_real, fourier_imag)
        )

        CQT = torch.stack((CQT_real, -CQT_imag), -1)

        if normalization_type == "librosa":
            CQT *= torch.sqrt(self.lenghts.view(-1, 1, 1)) / self.kernel_width
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT *= 2 / self.kernel_width
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        #         if self.norm:
        #             CQT = CQT/self.kernel_width*torch.sqrt(self.lenghts.view(-1,1,1))
        #         else:
        #             CQT = CQT*torch.sqrt(self.lenghts.view(-1,1,1))

        if output_format == "Magnitude":
            # Getting CQT Amplitude
            return torch.sqrt(CQT.pow(2).sum(-1))

        elif output_format == "Complex":
            return CQT

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT_imag, CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag, CQT_real))
            return torch.stack((phase_real, phase_imag), -1)

    def extra_repr(self) -> str:
        return "STFT kernel size = {}, CQT kernel size = {}".format(
            (*self.wcos.shape,), (*self.cqt_kernels_real.shape,)
        )


class CQT2010(nn.Module):
    """
    This algorithm is using the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave.
    Then we keep downsampling the input audio by a factor of 2 to convoluting it with the
    small CQT kernel. Everytime the input audio is downsampled, the CQT relative to the downsampled
    input is equavalent to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the code
    from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).
    early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.
    """

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        bins_per_octave=12,
        norm=True,
        basis_norm=1,
        window="hann",
        pad_mode="reflect",
        trainable_STFT=False,
        filter_scale=1,
        trainable_CQT=False,
        output_format="Magnitude",
        earlydownsample=True,
        verbose=True,
    ):

        super().__init__()

        self.norm = (
            norm  # Now norm is used to normalize the final CQT result by dividing n_fft
        )
        # basis_norm is for normalizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.output_format = output_format
        self.earlydownsample = (
            earlydownsample  # TODO: activate early downsampling later if possible
        )

        # This will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        # Creating lowpass filter and make it a torch tensor
        if verbose == True:
            print("Creating low pass filter ...", end="\r")
        start = time()
        lowpass_filter = torch.tensor(
            create_lowpass_filter(
                band_center=0.5, kernelLength=256, transitionBandwidth=0.001
            )
        )

        # Broadcast the tensor to the shape that fits conv1d
        self.register_buffer("lowpass_filter", lowpass_filter[None, None, :])

        if verbose == True:
            print(
                "Low pass filter created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )

        # Calculate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        # print("n_octaves = ", self.n_octaves)

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        # print("remainder = ", remainder)

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (
            1 - 1 / bins_per_octave
        )  # Adjusting the top minium bins
        if fmax_t > sr / 2:
            raise ValueError(
                "The top bin {}Hz has exceeded the Nyquist frequency, \
                              please reduce the n_bins".format(
                    fmax_t
                )
            )

        if (
            self.earlydownsample == True
        ):  # Do early downsampling if this argument is True
            if verbose == True:
                print("Creating early downsampling filter ...", end="\r")
            start = time()
            (
                sr,
                self.hop_length,
                self.downsample_factor,
                early_downsample_filter,
                self.earlydownsample,
            ) = get_early_downsample_params(
                sr, hop_length, fmax_t, Q, self.n_octaves, verbose
            )

            self.register_buffer("early_downsample_filter", early_downsample_filter)
            if verbose == True:
                print(
                    "Early downsampling filter created, \
                            time used = {:.4f} seconds".format(
                        time() - start
                    )
                )
        else:
            self.downsample_factor = 1.0

        # Preparing CQT kernels
        if verbose == True:
            print("Creating CQT kernels ...", end="\r")

        start = time()
        # print("Q = {}, fmin_t = {}, n_filters = {}".format(Q, self.fmin_t, n_filters))
        basis, self.n_fft, _, _ = create_cqt_kernels(
            Q,
            sr,
            self.fmin_t,
            n_filters,
            bins_per_octave,
            norm=basis_norm,
            topbin_check=False,
        )

        # This is for the normalization in the end
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs

        lenghts = np.ceil(Q * sr / freqs)
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer("lenghts", lenghts)

        self.basis = basis
        fft_basis = fft(basis)[
            :, : self.n_fft // 2 + 1
        ]  # Convert CQT kenral from time domain to freq domain

        # These cqt_kernel is already in the frequency domain
        cqt_kernels_real = torch.tensor(fft_basis.real)
        cqt_kernels_imag = torch.tensor(fft_basis.imag)

        if verbose == True:
            print(
                "CQT kernels created, time used = {:.4f} seconds".format(time() - start)
            )

        # print("Getting cqt kernel done, n_fft = ",self.n_fft)
        # Preparing kernels for Short-Time Fourier Transform (STFT)
        # We set the frequency range in the CQT filter instead of here.

        if verbose == True:
            print("Creating STFT kernels ...", end="\r")

        start = time()
        kernel_sin, kernel_cos, self.bins2freq, _, window = create_fourier_kernels(
            self.n_fft, window="ones", freq_scale="no"
        )
        wsin = kernel_sin * window
        wcos = kernel_cos * window

        wsin = torch.tensor(wsin)
        wcos = torch.tensor(wcos)

        if verbose == True:
            print(
                "STFT kernels created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )

        if trainable_STFT:
            wsin = nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter("wsin", wsin)
            self.register_parameter("wcos", wcos)
        else:
            self.register_buffer("wsin", wsin)
            self.register_buffer("wcos", wcos)

        if trainable_CQT:
            cqt_kernels_real = nn.Parameter(
                cqt_kernels_real, requires_grad=trainable_CQT
            )
            cqt_kernels_imag = nn.Parameter(
                cqt_kernels_imag, requires_grad=trainable_CQT
            )
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

            # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == "constant":
            self.padding = nn.ConstantPad1d(self.n_fft // 2, 0)
        elif self.pad_mode == "reflect":
            self.padding = nn.ReflectionPad1d(self.n_fft // 2)

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

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
        if self.earlydownsample == True:
            x = downsampling_by_n(
                x, self.early_downsample_filter, self.downsample_factor
            )
        hop = self.hop_length

        CQT = get_cqt_complex2(
            x,
            self.cqt_kernels_real,
            self.cqt_kernels_imag,
            hop,
            self.padding,
            wcos=self.wcos,
            wsin=self.wsin,
        )

        x_down = x  # Preparing a new variable for downsampling
        for i in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_2(x_down, self.lowpass_filter)

            CQT1 = get_cqt_complex2(
                x_down,
                self.cqt_kernels_real,
                self.cqt_kernels_imag,
                hop,
                self.padding,
                wcos=self.wcos,
                wsin=self.wsin,
            )
            CQT = torch.cat((CQT1, CQT), 1)

        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted top bins

        if normalization_type == "librosa":
            CQT *= torch.sqrt(self.lenghts.view(-1, 1, 1)) / self.n_fft
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT *= 2 / self.n_fft
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            # Getting CQT Amplitude
            return torch.sqrt(CQT.pow(2).sum(-1))

        elif output_format == "Complex":
            return CQT

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            phase_imag = torch.sin(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            return torch.stack((phase_real, phase_imag), -1)

    def extra_repr(self) -> str:
        return "STFT kernel size = {}, CQT kernel size = {}".format(
            (*self.wcos.shape,), (*self.cqt_kernels_real.shape,)
        )


class CQT1992v2(nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    This alogrithm uses the method proposed in [1]. I slightly modify it so that it runs faster
    than the original 1992 algorithm, that is why I call it version 2.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    filter_scale : float > 0
        Filter scale factor. Values of filter_scale smaller than 1 can be used to improve the time resolution at the
        cost of degrading the frequency resolution. Important to note is that setting for example filter_scale = 0.5 and
        bins_per_octave = 48 leads to exactly the same time-frequency resolution trade-off as setting filter_scale = 1
        and bins_per_octave = 24, but the former contains twice more frequency bins per octave. In this sense, values
        filter_scale < 1 can be seen to implement oversampling of the frequency axis, analogously to the use of zero
        padding when calculating the DFT.

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : string, float, or tuple
        The windowing function for CQT. If it is a string, It uses ``scipy.signal.get_window``. If it is a
        tuple, only the gaussian window wanrantees constant Q factor. Gaussian window should be given as a
        tuple ('gaussian', att) where att is the attenuation in the border given in dB.
        Please refer to scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``.

    output_format : str
        Determine the return type.
        ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``;
        ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``;
        ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT1992v2()
    >>> specs = spec_layer(x)
    """

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        bins_per_octave=12,
        filter_scale=1,
        norm=1,
        window="hann",
        center=True,
        pad_mode="reflect",
        trainable=False,
        output_format="Magnitude",
        verbose=True,
    ):

        super().__init__()

        self.trainable = trainable
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.output_format = output_format

        # creating kernels for CQT
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        if verbose == True:
            print("Creating CQT kernels ...", end="\r")

        start = time()
        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(
            Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax
        )

        self.register_buffer("lenghts", lenghts)
        self.frequencies = freqs

        cqt_kernels_real = torch.tensor(cqt_kernels.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag).unsqueeze(1)

        if trainable:
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

        if verbose == True:
            print(
                "CQT kernels created, time used = {:.4f} seconds".format(time() - start)
            )

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape

        normalization_type : str
            Type of the normalisation. The possible options are: \n
            'librosa' : the output fits the librosa one \n
            'convolutional' : the output conserves the convolutional inequalities of the wavelet transform:\n
            for all p ϵ [1, inf] \n
                - || CQT ||_p <= || f ||_p || g ||_1 \n
                - || CQT ||_p <= || f ||_1 || g ||_p \n
                - || CQT ||_2 = || f ||_2 || g ||_2 \n
            'wrap' : wraps positive and negative frequencies into positive frequencies. This means that the CQT of a
            sinus (or a cosinus) with a constant amplitude equal to 1 will have the value 1 in the bin corresponding to
            its frequency.
        """
        output_format = output_format or self.output_format

        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == "reflect":
                padding = nn.ReflectionPad1d(self.kernel_width // 2)

            x = padding(x)

        # CQT
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = -conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)

        if normalization_type == "librosa":
            CQT_real *= torch.sqrt(self.lenghts.view(-1, 1))
            CQT_imag *= torch.sqrt(self.lenghts.view(-1, 1))
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT_real *= 2
            CQT_imag *= 2
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            if self.trainable == False:
                # Getting CQT Amplitude
                CQT = torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2))
            else:
                CQT = torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2) + 1e-8)
            return CQT

        elif output_format == "Complex":
            return torch.stack((CQT_real, CQT_imag), -1)

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT_imag, CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag, CQT_real))
            return torch.stack((phase_real, phase_imag), -1)

    def forward_manual(self, x):
        """
        Method for debugging
        """

        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == "reflect":
                padding = nn.ReflectionPad1d(self.kernel_width // 2)

            x = padding(x)

        # CQT
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)

        # Getting CQT Amplitude
        CQT = torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2))
        return CQT * torch.sqrt(self.lenghts.view(-1, 1))


class CQT2010v2(nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    This alogrithm uses the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the
    input audio by a factor of 2 to convoluting it with the small CQT kernel.
    Everytime the input audio is downsampled, the CQT relative to the downsampled input is equivalent
    to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the
    code from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    Early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the
        argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically.
        Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    norm : bool
        Normalization for the CQT result.

    basis_norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``

    output_format : str
        Determine the return type.
        'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins, time_steps)``;
        'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins, time_steps, 2)``;
        'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT2010v2()
    >>> specs = spec_layer(x)
    """

    # To DO:
    # need to deal with the filter and other tensors

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
        window="hann",
        pad_mode="reflect",
        earlydownsample=True,
        trainable=False,
        output_format="Magnitude",
        verbose=True,
    ):

        super().__init__()

        self.norm = (
            norm  # Now norm is used to normalize the final CQT result by dividing n_fft
        )
        # basis_norm is for normalizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.earlydownsample = (
            earlydownsample  # We will activate early downsampling later if possible
        )
        self.trainable = trainable
        self.output_format = output_format

        # It will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        # Creating lowpass filter and make it a torch tensor
        if verbose == True:
            print("Creating low pass filter ...", end="\r")
        start = time()
        # self.lowpass_filter = torch.tensor(
        #                                     create_lowpass_filter(
        #                                     band_center = 0.50,
        #                                     kernelLength=256,
        #                                     transitionBandwidth=0.001))
        lowpass_filter = torch.tensor(
            create_lowpass_filter(
                band_center=0.50, kernelLength=256, transitionBandwidth=0.001
            )
        )

        # Broadcast the tensor to the shape that fits conv1d
        self.register_buffer("lowpass_filter", lowpass_filter[None, None, :])
        if verbose == True:
            print(
                "Low pass filter created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )

        # Caluate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        if verbose == True:
            print("num_octave = ", self.n_octaves)

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        # print("remainder = ", remainder)

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (
            1 - 1 / bins_per_octave
        )  # Adjusting the top minium bins
        if fmax_t > sr / 2:
            raise ValueError(
                "The top bin {}Hz has exceeded the Nyquist frequency, \
                            please reduce the n_bins".format(
                    fmax_t
                )
            )

        if (
            self.earlydownsample == True
        ):  # Do early downsampling if this argument is True
            if verbose == True:
                print("Creating early downsampling filter ...", end="\r")
            start = time()
            (
                sr,
                self.hop_length,
                self.downsample_factor,
                early_downsample_filter,
                self.earlydownsample,
            ) = get_early_downsample_params(
                sr, hop_length, fmax_t, Q, self.n_octaves, verbose
            )
            self.register_buffer("early_downsample_filter", early_downsample_filter)

            if verbose == True:
                print(
                    "Early downsampling filter created, \
                        time used = {:.4f} seconds".format(
                        time() - start
                    )
                )
        else:
            self.downsample_factor = 1.0

        # Preparing CQT kernels
        if verbose == True:
            print("Creating CQT kernels ...", end="\r")
        start = time()
        basis, self.n_fft, lenghts, _ = create_cqt_kernels(
            Q,
            sr,
            self.fmin_t,
            n_filters,
            bins_per_octave,
            norm=basis_norm,
            topbin_check=False,
        )
        # For normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs

        lenghts = np.ceil(Q * sr / freqs)
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer("lenghts", lenghts)

        self.basis = basis
        # These cqt_kernel is already in the frequency domain
        cqt_kernels_real = torch.tensor(basis.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(basis.imag).unsqueeze(1)

        if trainable:
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

        if verbose == True:
            print(
                "CQT kernels created, time used = {:.4f} seconds".format(time() - start)
            )
        # print("Getting cqt kernel done, n_fft = ",self.n_fft)

        # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == "constant":
            self.padding = nn.ConstantPad1d(self.n_fft // 2, 0)
        elif self.pad_mode == "reflect":
            self.padding = nn.ReflectionPad1d(self.n_fft // 2)

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

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
        if self.earlydownsample == True:
            x = downsampling_by_n(
                x, self.early_downsample_filter, self.downsample_factor
            )
        hop = self.hop_length
        CQT = get_cqt_complex(
            x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding
        )  # Getting the top octave CQT

        x_down = x  # Preparing a new variable for downsampling

        for i in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            CQT1 = get_cqt_complex(
                x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding
            )
            CQT = torch.cat((CQT1, CQT), 1)

        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted bottom bins
        # print("downsample_factor = ",self.downsample_factor)
        # print(CQT.shape)
        # print(self.lenghts.view(-1,1).shape)

        # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it
        # same mag as 1992
        CQT = CQT * self.downsample_factor
        # Normalize again to get same result as librosa
        if normalization_type == "librosa":
            CQT = CQT * torch.sqrt(self.lenghts.view(-1, 1, 1))
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT *= 2
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            if self.trainable == False:
                # Getting CQT Amplitude
                return torch.sqrt(CQT.pow(2).sum(-1))
            else:
                return torch.sqrt(CQT.pow(2).sum(-1) + 1e-8)

        elif output_format == "Complex":
            return CQT

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            phase_imag = torch.sin(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            return torch.stack((phase_real, phase_imag), -1)


class CQT(CQT1992v2):
    """An abbreviation for :func:`~nnAudio.Spectrogram.CQT1992v2`. Please refer to the :func:`~nnAudio.Spectrogram.CQT1992v2` documentation"""

    pass
