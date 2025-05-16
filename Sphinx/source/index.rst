.. nnAudio documentation master file, created by
   sphinx-quickstart on Tue Dec  3 10:57:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nnAudio |ProjectVersion|
===================================
Welcome to nnAudio |ProjectVersion|. A big shout out to `Miguel Pérez <https://github.com/migperfer>`_ who made this new update possible. Please feel free to check out his `github repositories <https://github.com/migperfer>`_ too.

This new version restructured the coding style, making things more modular and pythonic. In terms of functionalities, everything remains the same. In the future releases, ``nnAudio.Spectrogram`` will be replaced by ``nnAudio.features`` (see also :func:`~nnAudio.features`.)

:func:`~nnAudio.features.vqt.VQT` is finally avaliable in version |ProjectVersion| thanks to `Hao Hao Tan <https://github.com/gudgud96>`_!

Reminder: if you use nnAudio, please cite `The paper <https://ieeexplore.ieee.org/abstract/document/9174990>`_ describing its release.


Quick Start
***********
.. code-block:: python
    :emphasize-lines: 1,8-10,12

    from nnAudio import features
    from scipy.io import wavfile
    import torch
    sr, song = wavfile.read('./Bach.wav') # Loading your audio
    x = song.mean(1) # Converting Stereo  to Mono
    x = torch.tensor(x, device='cuda:0').float() # casting the array into a PyTorch Tensor

    spec_layer = features.STFT(n_fft=2048, freq_bins=None, hop_length=512, 
                                  window='hann', freq_scale='linear', center=True, pad_mode='reflect', 
                                  fmin=50,fmax=11025, sr=sr) # Initializing the model

    spec = spec_layer(x) # Feed-forward your waveform to get the spectrogram      

nnAudio is an audio processing toolbox using PyTorch convolutional neural
network as its backend. By doing so, spectrograms can be generated from
audio on-the-fly during neural network training and the Fourier kernels
(e.g. or CQT kernels) can be trained.
`Kapre <https://github.com/keunwoochoi/kapre>`__ has a similar concept
in which they also use 1D convolutional neural network to extract
spectrograms based on `Keras <https://keras.io>`__.

Other GPU audio processing tools are
`torchaudio <https://github.com/pytorch/audio>`__ and
`tf.signal <https://www.tensorflow.org/api_docs/python/tf/signal>`__.
But they are not using the neural network approach, and hence the
Fourier basis can not be trained. As of PyTorch 1.6.0, torchaudio is
still very difficult to install under the Windows environment due to
``sox``. nnAudio is a more compatible audio processing tool across
different operating systems since it relies mostly on PyTorch
convolutional neural network. The name of nnAudio comes from
``torch.nn``.

The implementation details for **nnAudio** have also been published in IEEE Access, people who are interested can read the `paper <https://ieeexplore.ieee.org/document/9174990>`__.

The source code for **nnAudio** can be found in `GitHub <https://github.com/KinWaiCheuk/nnAudio>`__.


.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    intro


.. toctree::
    :maxdepth: 1
    :caption: API Documentation
    
    nnAudio
 

.. toctree::
    :maxdepth: 1
    :caption: Examples
    
    examples
    
    
.. toctree::
    :maxdepth: 1
    :caption: GitHub
    
    github


.. toctree::
    :maxdepth: 1
    :caption: Citation
    
    citing


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
