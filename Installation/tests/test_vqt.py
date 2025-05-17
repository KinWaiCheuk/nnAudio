import pytest
import librosa
import torch
# import torchaudio
import sys

sys.path.insert(0, "./")

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

from nnAudio.features import CQT2010v2, VQT, CQT1992v2
import numpy as np
from parameters import *
import warnings

gpu_idx = 0  # Choose which GPU to use

# If GPU is avaliable, also test on GPU
if torch.cuda.is_available():
    device_args = ["cpu", f"cuda:{gpu_idx}"]
else:
    warnings.warn("GPU is not avaliable, testing only on CPU")
    device_args = ["cpu"]

# librosa example audio for testing
y, sr = librosa.load(librosa.ex('choice'), duration=5)

@pytest.mark.parametrize("device", [*device_args])
def test_vqt_gamma_zero(device):
    # print('input shape: ', y2.shape)
    # print('sr: ', sr)

    # nnAudio cqt
    spec = CQT1992v2(sr=sr, verbose=False).to(device)
    C2 = spec(torch.tensor(y).unsqueeze(0).to(device), output_format="Magnitude", normalization_type='librosa')
    C2 = C2.cpu().numpy().squeeze()

    # nnAudio vqt
    spec = VQT(sr=sr, gamma=0, verbose=False).to(device)
    V2 = spec(torch.tensor(y).unsqueeze(0).to(device), output_format="Magnitude", normalization_type='librosa')
    V2 = V2.cpu().numpy().squeeze()

    ## FOR TESTING DIFFERENTS PARAMETERS PURPOSES ONLY

    # # nnAudio cqt
    # spec = CQT1992v2(sr=sr, verbose=False, hop_length=441, n_bins=128, bins_per_octave=24).to(device)
    # C2 = spec(torch.tensor(y).unsqueeze(0).to(device), output_format="Magnitude", normalization_type='librosa')
    # C2 = C2.cpu().numpy().squeeze()

    # # nnAudio vqt
    # spec = VQT(sr=sr, gamma=0, verbose=False, hop_length=441, n_bins=128, bins_per_octave=24).to(device)
    # V2 = spec(torch.tensor(y).unsqueeze(0).to(device), output_format="Magnitude", normalization_type='librosa')
    # V2 = V2.cpu().numpy().squeeze()

    # check mel spect size
    # spec = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=22050,
    #     n_fft=1024,
    #     hop_length=441,
    #     f_min=30,
    #     f_max=11000,
    #     n_mels=128,
    #     mel_scale="slaney",
    #     normalized="frame_length",
    #     power=1,
    # ).to(device)
    # M2 = spec(torch.tensor(y).unsqueeze(0).to(device))
    # M2 = M2.cpu().numpy().squeeze()

    # print('C2 shape: ', C2.shape)
    # print('V2 shape: ', V2.shape)
    # print('V2 shape: ', M2.shape)
    # print('successfully alter vqt hop length')

    assert (C2 == V2).all() == True


@pytest.mark.parametrize("device", [*device_args])
def test_vqt(device):
    # print('second test')
    for gamma in [0, 1, 2, 5, 10]:

        # librosa vqt
        V1 = np.abs(librosa.vqt(y, sr=sr, gamma=gamma))

        # nnAudio vqt
        spec = VQT(sr=sr, gamma=gamma, verbose=False).to(device)
        V2 = spec(torch.tensor(y).unsqueeze(0).to(device), output_format="Magnitude", normalization_type='librosa')
        V2 = V2.cpu().numpy().squeeze()

        # NOTE: there will still be some diff between librosa and nnAudio vqt values (same as cqt)
        # mainly due to the lengths of both - librosa uses float but nnAudio uses int
        # this test aims to keep the diff range within a baseline threshold
        vqt_diff = np.abs(V1 - V2)

        # print('v1 shape: ', V1.shape)
        # print('v2 shape: ', V2.shape)
        assert np.allclose(V1,V2,1e-3,0.8)