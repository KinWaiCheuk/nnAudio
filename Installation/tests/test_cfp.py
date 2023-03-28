import pytest
import librosa
import torch
from scipy.signal import chirp, sweep_poly
import sys

sys.path.insert(0, "./")

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

from nnAudio.Spectrogram import *
from parameters import *
import warnings

episilon = 1
gpu_idx = 0  # Choose which GPU to use

# If GPU is avaliable, also test on GPU
if torch.cuda.is_available():
    device_args = ["cpu", f"cuda:{gpu_idx}"]
else:
    warnings.warn("GPU is not avaliable, testing only on CPU")
    device_args = ["cpu"]

# librosa example audio for testing
example_y, example_sr = librosa.load(librosa.example('vibeace', hq=False))


@pytest.mark.parametrize("device", [*device_args])
def test_cfp_original(device):
    x = torch.tensor(example_y, device=device).unsqueeze(0)

    cfp_layer = Combined_Frequency_Periodicity(
        fr=2,
        fs=44100,
        hop_length=320,
        window_size=2049,
        fc=80,
        tc=0.001,
        g=[0.24, 0.6, 1],
        NumPerOct=48,
    ).to(device)
    X = cfp_layer(x)
    ground_truth = torch.load(
        os.path.join(
            dir_path,
            "ground-truths/cfp_original.pt"),
        map_location=device
    )

    for i, j in zip(X, ground_truth):
        assert torch.allclose(torch.log(i+episilon),torch.log(j+episilon), 1e-3, 9e-1)

@pytest.mark.parametrize("device", [*device_args])
def test_cfp_new(device):
    x = torch.tensor(example_y, device=device).unsqueeze(0)

    cfp_layer = CFP(
        fr=2,
        fs=44100,
        hop_length=320,
        window_size=2049,
        fc=80,
        tc=0.001,
        g=[0.24, 0.6, 1],
        NumPerOct=48,
    ).to(device)
    X = cfp_layer(x)
    ground_truth = torch.load(
        os.path.join(dir_path, "ground-truths/cfp_new.pt"),
        map_location=device
    )
    assert torch.allclose(torch.log(X+episilon),torch.log(ground_truth+episilon), 1e-1, 2.5)


if torch.cuda.is_available():
    x = torch.randn((4, 44100)).to(
        f"cuda:{gpu_idx}"
    )  # Create a batch of input for the following Data.Parallel test

    @pytest.mark.parametrize("device", [f"cuda:{gpu_idx}"])
    def test_cfp_original_Parallel(device):
        cfp_layer = Combined_Frequency_Periodicity(
            fr=2,
            fs=44100,
            hop_length=320,
            window_size=2049,
            fc=80,
            tc=0.001,
            g=[0.24, 0.6, 1],
            NumPerOct=48,
        ).to(device)
        cfp_layer = torch.nn.DataParallel(cfp_layer)
        X = cfp_layer(x)

    @pytest.mark.parametrize("device", [f"cuda:{gpu_idx}"])
    def test_cfp_new_Parallel(device):
        cfp_layer = CFP(
            fr=2,
            fs=44100,
            hop_length=320,
            window_size=2049,
            fc=80,
            tc=0.001,
            g=[0.24, 0.6, 1],
            NumPerOct=48,
        ).to(device)
        X = cfp_layer(x.to(device))
