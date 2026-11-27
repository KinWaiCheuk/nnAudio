import pytest
import numpy as np
import scipy.signal
import torch
from scipy.signal import chirp
import sys

sys.path.insert(0, "./")
from nnAudio.features import MFCC, CFP, Combined_Frequency_Periodicity
import warnings

gpu_idx = 0  # Choose which GPU to use

# If GPU is avaliable, also test on GPU
if torch.cuda.is_available():
    device_args = ["cpu", f"cuda:{gpu_idx}"]
else:
    warnings.warn("GPU is not avaliable, testing only on CPU")
    device_args = ["cpu"]

sr = 22050
t = np.linspace(0, 1, sr, endpoint=False)
x = torch.tensor(chirp(t, f0=110, f1=4000, t1=1, method="logarithmic"), dtype=torch.float32)


def relative_error(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


@pytest.fixture
def blackmanharris(monkeypatch):
    # removed in SciPy 1.13 but still called by CFP's constructor
    if not hasattr(scipy.signal, "blackmanharris"):
        monkeypatch.setattr(scipy.signal, "blackmanharris",
                            scipy.signal.windows.blackmanharris, raising=False)


# full-scale audio overflows the float16 mel power spectrum, so float16 uses a quieter signal
mfcc_cases = [(torch.float16, 0.3, 3e-3), (torch.bfloat16, 1.0, 6e-2)]


@pytest.mark.parametrize("dtype, amplitude, tolerance", mfcc_cases)
@pytest.mark.parametrize("device", [*device_args])
def test_mfcc_half(device, dtype, amplitude, tolerance):
    signal = (x * amplitude).to(device).unsqueeze(0)
    ref = MFCC(sr=sr).to(device)(signal)
    out = MFCC(sr=sr).to(device).to(dtype)(signal.to(dtype))
    assert out.dtype == dtype and out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert relative_error(out, ref) < tolerance


# CFP computes in float32, so it must match float32 on the same half-rounded input and buffers
cfp_tolerance = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


def half_and_reference(cls, device, dtype):
    module = cls(fs=sr).to(device).to(dtype)
    reference = cls(fs=sr).to(device)
    reference.load_state_dict({k: v.float() for k, v in module.state_dict().items()})
    signal = x.to(device).to(dtype).unsqueeze(0)
    return module(signal), reference(signal.float())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", [*device_args])
def test_cfp_half(device, dtype, blackmanharris):
    out, ref = half_and_reference(CFP, device, dtype)
    assert out.dtype == dtype and out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert relative_error(out, ref) < cfp_tolerance[dtype]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", [*device_args])
def test_combined_frequency_periodicity_half(device, dtype, blackmanharris):
    outs, refs = half_and_reference(Combined_Frequency_Periodicity, device, dtype)
    assert len(outs) == len(refs) == 4
    for out, ref in zip(outs, refs):
        assert out.dtype == dtype and out.shape == ref.shape
        assert torch.isfinite(out).all()
        assert relative_error(out, ref) < cfp_tolerance[dtype]
