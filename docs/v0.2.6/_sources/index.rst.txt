.. nnAudio documentation master file, created by
   sphinx-quickstart on Tue Dec  3 10:57:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nnAudio |ProjectVersion|
===================================
Welcome to nnAudio |ProjectVersion|. This new version changes the syntax of the spectrogram layers creation, 
such that ``stft_layer.to(device)`` can be used. This new version is more stable 
than the previous version since it is more compatible with other torch modules.

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
    :caption: Tutorials
    
    examples


.. toctree::
    :maxdepth: 1
    :caption: Citation
    
    citing


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
