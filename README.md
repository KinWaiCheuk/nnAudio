# nnAudio
nnAudio is an audio processing toolbox using PyTorch convolutional neural network as its backend. By doing so, spectrograms can be generated from audio on-the-fly during neural network training and the Fourier kernels (e.g. or CQT kernels) can be trained. [Kapre](https://github.com/keunwoochoi/kapre) has a similar concept in which they also use 1D convolutional neural network to extract spectrograms based on [Keras](https://keras.io).

Other GPU audio processing tools are [torchaudio](https://github.com/pytorch/audio) and [tf.signal](https://www.tensorflow.org/api_docs/python/tf/signal). But they are not using the neural network approach, and hence the Fourier basis can not be trained. As of PyTorch 1.6.0, torchaudio is still very difficult to install under the Windows environment due to `sox`. nnAudio is a more compatible audio processing tool across different operating systems since it relies mostly on PyTorch convolutional neural network. The name of nnAudio comes from `torch.nn`

## Installation
`pip install git+https://github.com/KinWaiCheuk/nnAudio.git#subdirectory=Installation`

or

`pip install nnAudio==0.3.1`

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
| VQT | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Gammatone | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| CFP<sup>1</sup> | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GPU support | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

✅: Fully support    ☑️: Developing (only available in dev version)    ❌: Not support

<sup>1</sup> [Combining Spectral and Temporal Representations for Multipitch Estimation of Polyphonic Music](https://ieeexplore.ieee.org/document/7118691)

## News & Changelog
To view the full changelog, please go to [CHANGELOG.md](CHANGELOG.md)

**version 0.3.1** (24 Dec 2021):
1. Added VQT feature [#113](/../../pull/113)

**version 0.3.0** (19 Nov 2021):
1. Changed module naming. `nnAudio.Spectrogram` will be replaced by `nnAudio.features` in the future releases. Currently, various spectrogram types are accessible via both methods.


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


(Quick tips for unit test: `cd` inside Installation folder, then type `pytest`. You need at least 1931 MiB GPU memory to pass all the unit tests)

Alternatively, you may also contribute by:
   1. Making a better demonstration code or tutorial




## Dependencies
Numpy >= 1.14.5

Scipy >= 1.2.0

PyTorch >= 1.6.0 (Griffin-Lim only available after 1.6.0)

Python >= 3.6

librosa = 0.7.0 (Theoretically nnAudio depends on librosa. But we only need to use a single function `mel` from `librosa.filters`. To save users troubles from installing librosa for this single function, I just copy the chunk of functions corresponding to `mel` in my code so that nnAudio runs without the need to install librosa)



## Other similar libraries
[Kapre](https://www.semanticscholar.org/paper/Kapre%3A-On-GPU-Audio-Preprocessing-Layers-for-a-of-Choi-Joo/b1ad5643e5dd66fac27067b00e5c814f177483ca?citingPapersSort=is-influential#citing-papers)

[torch-stft](https://github.com/pseeth/torch-stft)


