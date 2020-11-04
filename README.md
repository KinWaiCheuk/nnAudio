# nnAudio
nnAudio is an audio processing toolbox using PyTorch convolutional neural network as its backend. By doing so, spectrograms can be generated from audio on-the-fly during neural network training and the Fourier kernels (e.g. or CQT kernels) can be trained. [Kapre](https://github.com/keunwoochoi/kapre) has a similar concept in which they also use 1D convolutional neural network to extract spectrograms based on [Keras](https://keras.io).

Other GPU audio processing tools are [torchaudio](https://github.com/pytorch/audio) and [tf.signal](https://www.tensorflow.org/api_docs/python/tf/signal). But they are not using the neural network approach, and hence the Fourier basis can not be trained. As of PyTorch 1.6.0, torchaudio is still very difficult to install under the Windows environment due to `sox`. nnAudio is a more compatible audio processing tool across different operation systems since it relies mostly on PyTorch convolutional neural network. The name of nnAudio comes from `torch.nn`

## Documentation
https://kinwaicheuk.github.io/nnAudio/index.html

## Call for Contributions
nnAudio is a fast growing package. With increasing number of feature requests, we welcome anyone who is familiar with digital signal processing and neural network to contribute to nnAudio. The current list of pending features includes:
1. Invertible Constant Q Transform (CQT)
1. CQT with filter scale factor (see issue [#54](/../../issues/54))
1. Variable Q Transform (see VQT[https://www.researchgate.net/publication/274009051_A_Matlab_Toolbox_for_Efficient_Perfect_Reconstruction_Time-Frequency_Transforms_with_Log-Frequency_Resolution])
1. Speed and Performance improvements for Griffin-Lim (see issue [#41](/../../issues/41))
1. Data Augmentation (see issue [#49](/../../issues/49))

(Quick tips for unit test: `cd` inside Installation folder, then type `pytest`. You need at least 1931 MiB GPU memory to pass all the unit tests)

Alternatively, you may also contribute by:
   1. Refactoring the code strucutre (Now all functions are within the same file, but with increasing number of features, I think we need to break it down into smaller modules)
   1. Making a better demonstration code or tutorial
   1. Improving the documentation page (It is ugly and out-dated at the most. Probably a documentation page with tutorial, if point 2 is finished before this)



## News & Changelog
**version 0.1.5**: Change
Much better `iSTFT` and `Griffin-Lim`. Now Griffin-Lim is a seperated PyTorch class and requires `torch >= 1.6.0` to run. `STFT` has also been refactored and it is less memory consuming now.
To use this version, do `pip install nnAudio==0.1.5`.

**version 0.1.4a0**: Finalized `iSTFT` and `Griffin-Lim`. They are now more accurate and stable.
To use this version, do `pip install nnAudio==0.1.4a0`.

**version 0.1.2.dev3**: Add `win_length` to `STFT` so that it has the same funcationality as librosa.\
To use this version, do `pip install nnAudio --pre -U`.

**version 0.1.2.dev2**: Fix bugs where the inverse cannot be done using GPU. And add a seperated `iSTFT` layer class

**version 0.1.2.dev1**: Add Inverse STFT and Griffin-Lim. They are still under development, please use with care.
                    
**version 0.1.1**: Add MFCC (1 June 2020)

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
| GPU support | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

✅: Fully support    ☑️: Developing (only available in dev version)    ❌: Not support


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



# Dependencies
Numpy 1.14.5

Scipy 1.2.0

PyTorch >= 1.6.0 (Griffin-Lim only available after 1.6.0)

Python >= 3.6

librosa = 0.7.0 (Theortically nnAudio depends on librosa. But we only need to use a single function `mel` from `librosa.filters`. To save users troubles from installing librosa for this single function, I just copy the chunk of functions corresponding to `mel` in my code so that nnAudio runs without the need to install librosa)

# Usage
All the required codes and examples are inside the jupyter-notebook. The audio processing layer can be integrated as part of the neural network as shown below. The [demo](https://colab.research.google.com/drive/1Zuf0vIFjvmHFbKjw4YOpALswc7A33UGK) on colab is also avaliable.

## Installation
To install stable version: `pip install nnAudio`

To install dev version: `pip install nnAudio --pre -U`.\
It allows the users to use the latest features, but the new features might not be stable. Please use with care and report any problems that you found.

## Standalone Usage
To use nnAudio, you need to define the neural network layer. After that, you can pass a batch of waveform to that layer to obtain the spectrograms. The input shape should be `(batch, len_audio)`.

```python
from nnAudio import Spectrogram
from scipy.io import wavfile
import torch
sr, song = wavfile.read('./Bach.wav') # Loading your audio
x = song.mean(1) # Converting Stereo  to Mono
x = torch.tensor(x).float() # casting the array into a PyTorch Tensor

spec_layer = Spectrogram.STFT(n_fft=2048, freq_bins=None, hop_length=512, 
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect', 
                              fmin=50,fmax=11025, sr=sr) # Initializing the model
                              
spec = spec_layer(x) # Feed-forward your waveform to get the spectrogram                                                        
```

## On-the-fly audio processing
One application for nnAudio is on-the-fly spectrogram generation when integrating it inside your neural network
```diff
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Getting Mel Spectrogram on the fly
+       self.spec_layer = Spectrogram.STFT(n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050, trainable=False, output_format='Magnitude', device='cuda:0')
        self.n_bins = freq_bins         
        
        # Creating CNN Layers
        self.CNN_freq_kernel_size=(128,1)
        self.CNN_freq_kernel_stride=(2,1)
        k_out = 128
        k2_out = 256
        self.CNN_freq = nn.Conv2d(1,k_out,
                                kernel_size=self.CNN_freq_kernel_size,stride=self.CNN_freq_kernel_stride)
        self.CNN_time = nn.Conv2d(k_out,k2_out,
                                kernel_size=(1,regions),stride=(1,1))    
        
        self.region_v = 1 + (self.n_bins-self.CNN_freq_kernel_size[0])//self.CNN_freq_kernel_stride[0]
        self.linear = torch.nn.Linear(k2_out*self.region_v, m, bias=False)
        
    def forward(self,x):
+        z = self.spec_layer(x)
        z = torch.log(z+epsilon)
        z2 = torch.relu(self.CNN_freq(z.unsqueeze(1)))
        z3 = torch.relu(self.CNN_time(z2))
        y = self.linear(torch.relu(torch.flatten(z3,1)))
        return torch.sigmoid(y)
```
## Trainable Fourier basis
Fourier basis in `STFT()` can be set trainable by using `trainable=True` argument. Fourier basis in `MelSpectrogram()` can be set trainable by using `trainable_STFT=True`, and Mel filter banks can be set trainable using `trainable_mel=False` argument.

The follow demonstrations are avaliable on Google colab.

1. [Trainable STFT Kernel](https://colab.research.google.com/drive/12VwjKSuXFkXCQd1hr3KUZ2bqzFEe-O6L)
1. [Trainable Mel Kernel](https://colab.research.google.com/drive/1UtswBYWhVxDNBRDajWzyplZfMiqENCEF)
1. [Trainable CQT Kernel](https://colab.research.google.com/drive/1coH54dfjAOxEyOjJrqscQRyC0_lmF04s)

The figure below shows the STFT basis before and after training.

![alt text](https://github.com/KinWaiCheuk/nnAudio/blob/master/Trainable_STFT/Trained_basis.png)

The figure below shows how is the STFT output affected by the changes in STFT basis. Notice the subtle signal in the background for the trained STFT.

<img src="https://github.com/KinWaiCheuk/nnAudio/blob/master/figures/STFT_training.png" :height="83%" width="83%">

## Using GPU
If GPU is avaliable in your computer, you should put the following command at the beginning of your script to ensure nnAudio is run in GPU. By default, PyTorch tensors are created in CPU, if you want to use nnAudio in GPU, make sure to transfer all your PyTorch tensors to GPU
```python
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
```

Then you can initialize nnAudio by choosing either CPU or GPU with the `device` argument. The default setting for nnAudio is `device='cuda:0'`

```python
spec_layer = Spectrogram.STFT(device=device)
```

## Functionalities
Currently there are 4 models to generate various types of spectrograms.
### 1. STFT
```python
from nnAudio import Spectrogram
Spectrogram.STFT(n_fft=2048, freq_bins=None, hop_length=512, window='hann', freq_scale='no', center=True, pad_mode='reflect', fmin=50,fmax=6000, sr=22050, trainable=False)
```

```
freq_scale: 'no', 'linear', or 'log'. This options controls the spacing of frequency among Fourier basis. When chosing 'no', the STFT output is same as the librosa output. fmin and fmax will have no effect under this option. When chosing 'linear' or 'log', the frequency scale will be in linear scale or logarithmic scale with the
```

### 2. Mel Spectrogram
```python
Spectrogram.MelSpectrogram(sr=22050, n_fft=2048, n_mels=128, hop_length=512, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
```

### 3. CQT
```python
Spectrogram.CQT(sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
```
### 4. MFCC
```python
Spectrogram.MFCC(sr=22050, n_mfcc=20, norm='ortho', device='cuda:0', verbose=True, **kwargs)
```

### Differences between CQT1992 and CQT2010
The result for CQT1992 is smoother than CQT2010 and librosa. Since librosa and CQT2010 are using the same algorithm (downsampling approach as mentioned in this [paper](https://www.semanticscholar.org/paper/CONSTANT-Q-TRANSFORM-TOOLBOX-FOR-MUSIC-PROCESSING-Sch%C3%B6rkhuber/4cef10ea66e40ad03f434c70d745a4959cea96dd)), you can see similar artifacts as a result of downsampling. The default `CQT` in nnAudio is the 1992 version, with slighltly modifications to make it faster than the original CQT proposed in [1992](https://www.semanticscholar.org/paper/An-efficient-algorithm-for-the-calculation-of-a-Q-Brown-Puckette/628a0981e2ed1c33b1b9a88018a01e2f0be0c956).

However, all both versions of CQT are avaliable for users to choose. To explicitly choose which CQT to use, you can call nnAudio with the following code.

#### CQT1992
It is the default algorithm used in `Spectrogram.CQT`.
```python
Spectrogram.CQT1992v2(sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
```

#### CQT2010
```python
Spectrogram.CQT2010v2(sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
```

![alt text](https://github.com/KinWaiCheuk/nnAudio/blob/master/figures/CQT_compare.png)


## Output Spectrogram
The spectrogram outputs from nnAudio are nearly identical to the implmentation of librosa. Four different input singals, linear sine sweep, logarithmic sine sweep, impluse, and piano chromatic scale, are used to test the nnAudio output. The figures below shows the result.

![alt text](https://github.com/KinWaiCheuk/nnAudio/blob/master/figures/performance_1.png)
![alt text](https://github.com/KinWaiCheuk/nnAudio/blob/master/figures/performance_2.png)


## Speed
The speed test is conducted using three different machines

Machine A: Windows Desktop with CPU: Intel Core i7-8700 @ 3.20GHz and GeForce GTX 1070 Ti 8Gb GPU

Machine B: Linux Desktop with CPU: AMD Ryzen 7 PRO 3700 and 1 GeForce RTX 2080 Ti 11Gb GPU

Machine C: DGX station with CPU: Intel Xeon E5-2698 v4 @ 2.20GHz and Tesla v100 32Gb GPU

![alt text](https://github.com/KinWaiCheuk/nnAudio/blob/master/figures/speedv3.png)

  

}

# Other similar libraries
[Kapre](https://www.semanticscholar.org/paper/Kapre%3A-On-GPU-Audio-Preprocessing-Layers-for-a-of-Choi-Joo/b1ad5643e5dd66fac27067b00e5c814f177483ca?citingPapersSort=is-influential#citing-papers)

[torch-stft](https://github.com/pseeth/torch-stft)




