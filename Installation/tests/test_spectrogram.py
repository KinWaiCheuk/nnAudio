import pytest
import librosa
import torch
import matplotlib.pyplot as plt
from scipy.signal import chirp, sweep_poly
from nnAudio.Spectrogram import *
from parameters import *

gpu_idx=1

# librosa example audio for testing
example_y, example_sr = librosa.load(librosa.util.example_audio_file())


@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)  
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_inverse2(n_fft, hop_length, window, device):
    x = torch.tensor(example_y,device=device)
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, device=device)
    istft = iSTFT(n_fft=n_fft, hop_length=hop_length, window=window, device=device)
    X = stft(x.unsqueeze(0), output_format="Complex")
    x_recon = istft(X, num_samples=x.shape[0]).squeeze()
    assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-3, atol=1)    

@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_inverse(n_fft, hop_length, window, device):
    x = torch.tensor(example_y, device=device)
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, device=device)
    X = stft(x.unsqueeze(0), output_format="Complex")
    x_recon = stft.inverse(X, num_samples=x.shape[0]).squeeze()
    assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-3, atol=1)
    

    
# @pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)

# def test_inverse_GPU(n_fft, hop_length, window):
#     x = torch.tensor(example_y,device=f'cuda:{gpu_idx}')
#     stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, device=f'cuda:{gpu_idx}')
#     X = stft(x.unsqueeze(0), output_format="Complex")
#     x_recon = stft.inverse(X, num_samples=x.shape[0]).squeeze()
#     assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-3, atol=1)


@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_stft_complex(n_fft, hop_length, window, device):
    x = example_y
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, device=device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Complex")
    X_real, X_imag = X[:, :, :, 0].squeeze(), X[:, :, :, 1].squeeze()
    X_librosa = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=window)
    real_diff, imag_diff = np.allclose(X_real.cpu(), X_librosa.real, rtol=1e-3, atol=1e-3), \
                            np.allclose(X_imag.cpu(), X_librosa.imag, rtol=1e-3, atol=1e-3)
    
    assert real_diff and imag_diff 
    
# @pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)    
# def test_stft_complex_GPU(n_fft, hop_length, window):
#     x = example_y
#     stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, device=f'cuda:{gpu_idx}')
#     X = stft(torch.tensor(x,device=f'cuda:{gpu_idx}').unsqueeze(0), output_format="Complex")
#     X_real, X_imag = X[:, :, :, 0].squeeze().detach().cpu(), X[:, :, :, 1].squeeze().detach().cpu()
#     X_librosa = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=window)
#     real_diff, imag_diff = np.allclose(X_real, X_librosa.real, rtol=1e-3, atol=1e-3), \
#                             np.allclose(X_imag, X_librosa.imag, rtol=1e-3, atol=1e-3)
    
#     assert real_diff and imag_diff        
    
@pytest.mark.parametrize("n_fft, win_length, hop_length", stft_with_win_parameters) 
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_stft_complex_winlength(n_fft, win_length, hop_length, device):
    x = example_y
    stft = STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length, device=device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Complex")
    X_real, X_imag = X[:, :, :, 0].squeeze(), X[:, :, :, 1].squeeze()
    X_librosa = librosa.stft(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    real_diff, imag_diff = np.allclose(X_real.cpu(), X_librosa.real, rtol=1e-3, atol=1e-3), \
                            np.allclose(X_imag.cpu(), X_librosa.imag, rtol=1e-3, atol=1e-3)
    assert real_diff and imag_diff    
              
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_stft_magnitude(device):
    x = example_y
    stft = STFT(n_fft=2048, hop_length=512, device=device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Magnitude").squeeze()
    X_librosa, _ = librosa.core.magphase(librosa.stft(x, n_fft=2048, hop_length=512))
    assert np.allclose(X.cpu(), X_librosa, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_stft_phase(device):
    x = example_y
    stft = STFT(n_fft=2048, hop_length=512, device=device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Phase")
    X_real, X_imag = torch.cos(X).squeeze(), torch.sin(X).squeeze()
    _, X_librosa = librosa.core.magphase(librosa.stft(x, n_fft=2048, hop_length=512))

    real_diff, imag_diff = np.mean(np.abs(X_real.cpu().numpy() - X_librosa.real)), \
                            np.mean(np.abs(X_imag.cpu().numpy() - X_librosa.imag))

    # I find that np.allclose is too strict for allowing phase to be similar to librosa.
    # Hence for phase we use average element-wise distance as the test metric.
    assert real_diff < 2e-4 and imag_diff < 2e-4

@pytest.mark.parametrize("n_fft, win_length", mel_win_parameters)  
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_mel_spectrogram(n_fft, win_length, device):
    x = example_y
    melspec = MelSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=512, device=device)
    X = melspec(torch.tensor(x, device=device).unsqueeze(0)).squeeze()
    X_librosa = librosa.feature.melspectrogram(x, n_fft=n_fft, win_length=win_length, hop_length=512)
    assert np.allclose(X.cpu(), X_librosa, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_1992_v2_log(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='logarithmic')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT1992v2(sr=fs, fmin=55, device=device, output_format="Magnitude",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-1992-mag-ground-truth.npy")
    X = torch.log(X + 1e-5)
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT1992v2(sr=fs, fmin=55, device=device, output_format="Complex",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-1992-complex-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
    stft = CQT1992v2(sr=fs, fmin=55, device=device, output_format="Phase",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-1992-phase-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_1992_v2_linear(device):
    # Linear sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='linear')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT1992v2(sr=fs, fmin=55, device=device, output_format="Magnitude",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-1992-mag-ground-truth.npy")
    X = torch.log(X + 1e-5)
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT1992v2(sr=fs, fmin=55, device=device, output_format="Complex",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-1992-complex-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
    stft = CQT1992v2(sr=fs, fmin=55, device=device, output_format="Phase",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-1992-phase-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_2010_v2_log(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='logarithmic')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT2010v2(sr=fs, fmin=55, device=device, output_format="Magnitude",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-2010-mag-ground-truth.npy")
    X = torch.log(X + 1e-5)
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT2010v2(sr=fs, fmin=55, device=device, output_format="Complex",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-2010-complex-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
    stft = CQT2010v2(sr=fs, fmin=55, device=device, output_format="Phase",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-2010-phase-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_2010_v2_linear(device):
    # Linear sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='linear')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT2010v2(sr=fs, fmin=55, device=device, output_format="Magnitude",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-2010-mag-ground-truth.npy")
    X = torch.log(X + 1e-5)
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT2010v2(sr=fs, fmin=55, device=device, output_format="Complex",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-2010-complex-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
    stft = CQT2010v2(sr=fs, fmin=55, device=device, output_format="Phase",
                     n_bins=207, bins_per_octave=24)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-2010-phase-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_mfcc(device):
    x = example_y
    mfcc = MFCC(device=device, sr=example_sr)
    X = mfcc(torch.tensor(x, device=device).unsqueeze(0)).squeeze()
    X_librosa = librosa.feature.mfcc(x, sr=example_sr)
    assert np.allclose(X.cpu(), X_librosa, rtol=1e-3, atol=1e-3)

