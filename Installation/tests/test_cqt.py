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
def test_cqt_1992(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs * t)
    x = chirp(s, f0, 1, f1, method="logarithmic")
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT1992(
        sr=fs, fmin=220, output_format="Magnitude", n_bins=80, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))

    # Complex
    stft = CQT1992(
        sr=fs, fmin=220, output_format="Complex", n_bins=80, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))

    # Phase
    stft = CQT1992(
        sr=fs, fmin=220, output_format="Phase", n_bins=160, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))

    assert True


@pytest.mark.parametrize("device", [*device_args])
def test_cqt_2010(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs * t)
    x = chirp(s, f0, 1, f1, method="logarithmic")
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT2010(
        sr=fs, fmin=110, output_format="Magnitude", n_bins=160, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))

    # Complex
    stft = CQT2010(
        sr=fs, fmin=110, output_format="Complex", n_bins=160, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))

    # Phase
    stft = CQT2010(
        sr=fs, fmin=110, output_format="Phase", n_bins=160, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    assert True


@pytest.mark.parametrize("device", [*device_args])
def test_cqt_1992_v2_log(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs * t)
    x = chirp(s, f0, 1, f1, method="logarithmic")
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT1992v2(
        sr=fs, fmin=55, output_format="Magnitude", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load(
        os.path.join(dir_path, "ground-truths/log-sweep-cqt-1992-mag-ground-truth.npy")
    )
    X = torch.log(X + 1e-5)
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT1992v2(
        sr=fs, fmin=55, output_format="Complex", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load(
        os.path.join(
            dir_path, "ground-truths/log-sweep-cqt-1992-complex-ground-truth.npy"
        )
    )
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
    stft = CQT1992v2(
        sr=fs, fmin=55, output_format="Phase", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load(
        os.path.join(
            dir_path, "ground-truths/log-sweep-cqt-1992-phase-ground-truth.npy"
        )
    )
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", [*device_args])
def test_cqt_1992_v2_linear(device):
    # Linear sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs * t)
    x = chirp(s, f0, 1, f1, method="linear")
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT1992v2(
        sr=fs, fmin=55, output_format="Magnitude", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load(
        os.path.join(
            dir_path, "ground-truths/linear-sweep-cqt-1992-mag-ground-truth.npy"
        )
    )
    X = torch.log(X + 1e-5)
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT1992v2(
        sr=fs, fmin=55, output_format="Complex", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load(
        os.path.join(
            dir_path, "ground-truths/linear-sweep-cqt-1992-complex-ground-truth.npy"
        )
    )
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
    stft = CQT1992v2(
        sr=fs, fmin=55, output_format="Phase", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load(
        os.path.join(
            dir_path, "ground-truths/linear-sweep-cqt-1992-phase-ground-truth.npy"
        )
    )
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", [*device_args])
def test_cqt_2010_v2_log(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs * t)
    x = chirp(s, f0, 1, f1, method="logarithmic")
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT2010v2(
        sr=fs, fmin=55, output_format="Magnitude", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    X = torch.log(X + 1e-2)
    #     np.save(os.path.join(dir_path, "ground-truths/log-sweep-cqt-2010-mag-ground-truth", X.cpu())
    ground_truth = np.load(
        os.path.join(dir_path, "ground-truths/log-sweep-cqt-2010-mag-ground-truth.npy")
    )
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT2010v2(
        sr=fs, fmin=55, output_format="Complex", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    #     np.save(os.path.join(dir_path, "ground-truths/log-sweep-cqt-2010-complex-ground-truth", X.cpu())
    ground_truth = np.load(
        os.path.join(
            dir_path, "ground-truths/log-sweep-cqt-2010-complex-ground-truth.npy"
        )
    )
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", [*device_args])
def test_cqt_2010_v2_linear(device):
    # Linear sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs * t)
    x = chirp(s, f0, 1, f1, method="linear")
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT2010v2(
        sr=fs, fmin=55, output_format="Magnitude", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    X = torch.log(X + 1e-2)
    #     np.save(os.path.join(dir_path, "ground-truths/linear-sweep-cqt-2010-mag-ground-truth", X.cpu())
    ground_truth = np.load(
        os.path.join(
            dir_path, "ground-truths/linear-sweep-cqt-2010-mag-ground-truth.npy"
        )
    )
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT2010v2(
        sr=fs, fmin=55, output_format="Complex", n_bins=207, bins_per_octave=24
    ).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    #     np.save(os.path.join(dir_path, "ground-truths/linear-sweep-cqt-2010-complex-ground-truth", X.cpu())
    ground_truth = np.load(
        os.path.join(
            dir_path, "ground-truths/linear-sweep-cqt-2010-complex-ground-truth.npy"
        )
    )
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)


if torch.cuda.is_available():
    x = torch.randn((4, 44100)).to(
        f"cuda:{gpu_idx}"
    )  # Create a batch of input for the following Data.Parallel test

    @pytest.mark.parametrize("device", [f"cuda:{gpu_idx}"])
    def test_CQT1992_Parallel(device):
        spec_layer = CQT1992(fmin=110, n_bins=60, bins_per_octave=12).to(device)
        spec_layer_parallel = torch.nn.DataParallel(spec_layer)
        spec = spec_layer_parallel(x)

    @pytest.mark.parametrize("device", [f"cuda:{gpu_idx}"])
    def test_CQT1992v2_Parallel(device):
        spec_layer = CQT1992v2().to(device)
        spec_layer_parallel = torch.nn.DataParallel(spec_layer)
        spec = spec_layer_parallel(x)

    @pytest.mark.parametrize("device", [f"cuda:{gpu_idx}"])
    def test_CQT2010_Parallel(device):
        spec_layer = CQT2010().to(device)
        spec_layer_parallel = torch.nn.DataParallel(spec_layer)
        spec = spec_layer_parallel(x)

    @pytest.mark.parametrize("device", [f"cuda:{gpu_idx}"])
    def test_CQT2010v2_Parallel(device):
        spec_layer = CQT2010v2().to(device)
        spec_layer_parallel = torch.nn.DataParallel(spec_layer)
        spec = spec_layer_parallel(x)
