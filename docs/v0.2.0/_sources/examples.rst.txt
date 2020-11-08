Tutorials
=============

Call for Contribution:
**********************


nnAudio is a fast-growing package. With the increasing number of feature requests, we welcome anyone who is familiar with digital signal processing and neural network to contribute to nnAudio. The current list of pending features includes:

1. Invertible Constant Q Transform (CQT)
2. CQT with filter scale factor (see issue `#54 <https://github.com/KinWaiCheuk/nnAudio/issues/54>`__)
3. Variable Q Transform see `VQT <https://www.researchgate.net/publication/274009051_A_Matlab_Toolbox_for_Efficient_Perfect_Reconstruction_Time-Frequency_Transforms_with_Log-Frequency_Resolution>`__)
4. Speed and Performance improvements for Griffin-Lim (see issue `#41 <https://github.com/KinWaiCheuk/nnAudio/issues/41>`__)
5. Data Augmentation (see issue `#49 <https://github.com/KinWaiCheuk/nnAudio/issues/49>`__)

(Quick tips for unit test: `cd` inside Installation folder, then type `pytest`. You need at least 1931 MiB GPU memory to pass all the unit tests)

Alternatively, you may also contribute by:

1. Refactoring the code structure (Now all functions are within the same file, but with the increasing number of features, I think we need to break it down into smaller modules)
2. Making a better demonstration code or tutorial

People who are interested in contributing to nnAudio can visit
the `github page <https://github.com/KinWaiCheuk/nnAudio>`_ or 
contact me via kinwai<underscore>cheuk<at>mymail.sutd.edu.sg.


