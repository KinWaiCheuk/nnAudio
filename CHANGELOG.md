# Changelog
**version 0.2.3** (10 June 2021): 
1. CQT2010 bug has been fixed [#85](/../../issues/85).
1. Provide a wider support for scipy versions using `from scipy.fftpack import fft` in [utils.py](https://github.com/KinWaiCheuk/nnAudio/blob/e9b1697963f0fd8e5030b130a30974bc06408baf/Installation/nnAudio/utils.py#L13)
1. Documentation error for STFT has been fixed [#90](/../../issues/90)

This version can be obtained via:

`pip install git+https://github.com/KinWaiCheuk/nnAudio.git#subdirectory=Installation`.

or

`pip install nnAudio==0.2.3`

**version 0.2.2** (1 March 2021): 
Added filter scale support to various version of CQT classes as requested in [#54](/../../issues/54). Different normalization methods are also added to the `forward()` method as `normalization_type` under each CQT class. A bug is discovered in CQT2010, the output is problematic [#85](/../../issues/85).

To use this version, do `pip install nnAudio==0.2.2`.

**version 0.2.1** (15 Jan 2021): 
Fixed bugs [#80](/../../issues/80), [#82](/../../issues/82), and fulfilled request [#83](/../../issues/83). nnAudio version can be checked with `nnAudio.__version__` inside python now. Added two more spectrogram types `Gammatonegram()` and `Combined_Frequency_Periodicity()`.

To use this version, do `pip install nnAudio==0.2.1`.

**version 0.2.0** (8 Nov 2020): 
Now it is possible to do `stft_layer.to(device)` to move the spectrogram layers between different devices.
No more `device` argument when creating the spectrogram layers.

To use this version, do `pip install nnAudio==0.2.0`.

**version 0.1.5**:
Much better `iSTFT` and `Griffin-Lim`. Now Griffin-Lim is a separated PyTorch class and requires `torch >= 1.6.0` to run. `STFT` has also been refactored and it is less memory consuming now.

To use this version, do `pip install nnAudio==0.1.5`.

**version 0.1.4a0**: Finalized `iSTFT` and `Griffin-Lim`. They are now more accurate and stable.

**version 0.1.2.dev3**: Add `win_length` to `STFT` so that it has the same funcationality as librosa.

**version 0.1.2.dev2**: Fix bugs where the inverse cannot be done using GPU. And add a separated `iSTFT` layer class

**version 0.1.2.dev1**: Add Inverse STFT and Griffin-Lim. They are still under development, please use with care.
                    
**version 0.1.1**  (1 June 2020): Add MFCC

