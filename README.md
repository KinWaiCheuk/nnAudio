# nnAudio
nnAudio is an audio processing toolbox using PyTorch convolutional neural network as its backend. By doing so, spectrograms can be generated from audio on-the-fly during neural network training and the Fourier kernels (e.g. or CQT kernels) can be trained. [Kapre](https://github.com/keunwoochoi/kapre) has a similar concept in which they also use 1D convolutional neural network to extract spectrograms based on [Keras](https://keras.io).

Other GPU audio processing tools are [torchaudio](https://github.com/pytorch/audio) and [tf.signal](https://www.tensorflow.org/api_docs/python/tf/signal). But they are not using the neural network approach, and hence the Fourier basis can not be trained. As of PyTorch 1.6.0, torchaudio is still very difficult to install under the Windows environment due to `sox`. nnAudio is a more compatible audio processing tool across different operating systems since it relies mostly on PyTorch convolutional neural network. The name of nnAudio comes from `torch.nn`


## Documentation
https://kinwaicheuk.github.io/nnAudio/index.html

## Comparison with other libraries
| Feature | [nnAudio](https://github.com/KinWaiCheuk/nnAudio) | [torch.stft](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SpectralOps.cpp) | [kapre](https://github.com/keunwoochoi/kapre) | [torchaudio](https://github.com/pytorch/audio) | [tf.signal](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/signal) | [torch-stft](https://github.com/pseeth/torch-stft) | [librosa](https://github.com/librosa/librosa) |
| ------- | ------- | ---------- | ----- | ---------- | ---------------------------- | ---------- | ------- |
| Trainable | ✅ | ❌| ✅ | ❌ | ❌ | ✅ | ❌ |
| Differentiable | ✅  | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Linear frequency STFT| ✅  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Logarithmic frequency STFT| ✅  | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Inverse STFT| ✅  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Griffin-Lim| ✅  | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| Mel | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| MFCC | ✅  | ❌ | ❌ | ✅| ✅ | ❌ | ✅ |
| CQT | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Gammatone | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| CFP<sup>1</sup> | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GPU support | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

✅: Fully support    ☑️: Developing (only available in dev version)    ❌: Not support

<sup>1</sup> [Combining Spectral and Temporal Representations for Multipitch Estimation of Polyphonic Music](https://ieeexplore.ieee.org/document/7118691)

## News & Changelog
**version 0.2.2** (1 March 2021): 
Added filter scale support to various version of CQT classes as requested in [#54](/../../issues/54). Different normalization methods are also added to the `forward()` method as `normalization_type` under each CQT class. A bug is discovered in CQT2010, the output is problematic [#85](/../../issues/85).

This version can be obtained via:
`pip install git+https://github.com/KinWaiCheuk/nnAudio.git#subdirectory=Installation`.

**version 0.2.1** (15 Jan 2021): 
Fixed bugs [#80](/../../issues/80), [#82](/../../issues/82), and fulfilled request [#83](/../../issues/83). nnAudio version can be checked with `nnAudio.__version__` inside python now. Added two more spectrogram types `Gammatonegram()` and `Combined_Frequency_Periodicity()`.

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




## How to cite nnAudio
The paper for nnAudio is avaliable on [IEEE Access](https://ieeexplore.ieee.org/document/9174990)

K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, "nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks," in IEEE Access, vol. 8, pp. 161981-162003, 2020, doi: 10.1109/ACCESS.2020.3019084.

### BibTex
@ARTICLE{9174990,
  author={K. W. {Cheuk} and H. {Anderson} and K. {Agres} and D. {Herremans}},
  journal={IEEE Access}, 
  title={nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks}, 
  year={2020},
  volume={8},
  number={},
  pages={161981-162003},
  doi={10.1109/ACCESS.2020.3019084}}


## Call for Contributions
nnAudio is a fast-growing package. With the increasing number of feature requests, we welcome anyone who is familiar with digital signal processing and neural network to contribute to nnAudio. The current list of pending features includes:
1. Invertible Constant Q Transform (CQT)
1. CQT with filter scale factor (see issue [#54](/../../issues/54))
1. Variable Q Transform (see VQT[https://www.researchgate.net/publication/274009051_A_Matlab_Toolbox_for_Efficient_Perfect_Reconstruction_Time-Frequency_Transforms_with_Log-Frequency_Resolution])
1. Speed and Performance improvements for Griffin-Lim (see issue [#41](/../../issues/41))
1. Data Augmentation (see issue [#49](/../../issues/49))

(Quick tips for unit test: `cd` inside Installation folder, then type `pytest`. You need at least 1931 MiB GPU memory to pass all the unit tests)

Alternatively, you may also contribute by:
   1. Refactoring the code structure (Now all functions are within the same file, but with the increasing number of features, I think we need to break it down into smaller modules)
   1. Making a better demonstration code or tutorial




## Dependencies
Numpy 1.14.5

Scipy 1.2.0

PyTorch >= 1.6.0 (Griffin-Lim only available after 1.6.0)

Python >= 3.6

librosa = 0.7.0 (Theoretically nnAudio depends on librosa. But we only need to use a single function `mel` from `librosa.filters`. To save users troubles from installing librosa for this single function, I just copy the chunk of functions corresponding to `mel` in my code so that nnAudio runs without the need to install librosa)



## Other similar libraries
[Kapre](https://www.semanticscholar.org/paper/Kapre%3A-On-GPU-Audio-Preprocessing-Layers-for-a-of-Choi-Joo/b1ad5643e5dd66fac27067b00e5c814f177483ca?citingPapersSort=is-influential#citing-papers)

[torch-stft](https://github.com/pseeth/torch-stft)
