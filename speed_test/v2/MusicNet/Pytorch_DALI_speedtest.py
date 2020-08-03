import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy.io import wavfile
import glob
from nnAudio import Spectrogram
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import time
import tqdm
import pickle
from pathlib import Path
import os

p = os.path.join(Path(__file__).parent,'train_data', '*.wav')

class MusicNet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
#         print(Path(__file__).parent)
        self.file_list = glob.glob(p)
        
    def __len__(self):
        return len(self.file_list)  
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_name = self.file_list[idx]
        sr, wav = wavfile.read(audio_name)

        return wav

if __name__ == '__main__':
    dataset = MusicNet()
    dataset = DataLoader(dataset, shuffle=False, num_workers=8)
    result = {}
    # STFT

    n_fft_ls = [256, 512, 1024, 2048, 4096]
    for n_fft in n_fft_ls:
        layer = Spectrogram.STFT(n_fft=n_fft, hop_length=512, verbose=False, device=device)
        start = time.time()
        for i in tqdm.tqdm(dataset):
            i = i.to(device)
            layer(i)
        result[f'STFT_{n_fft}'] = time.time()-start

    n_fft_ls = [256, 512, 1024, 2048, 4096]
    for n_fft in n_fft_ls:
        layer = Spectrogram.STFT(n_fft=n_fft, hop_length=512, freq_scale='log', sr=44100, fmin=1, fmax=22050, verbose=False, device=device)
        time.sleep(0.5)
        start = time.time()
        for i in tqdm.tqdm(dataset):
            i = i.to(device)
            layer(i)
        result[f'STFT-log_{n_fft}'] = time.time()-start


    # Mel

    n_fft_ls = [256, 512, 1024, 2048, 4096]
    for n_fft in n_fft_ls:
        n_mels_ls = [128, 256, 512, 1024, 2048]
        for n_mels in n_mels_ls:
            if n_mels < n_fft:
                layer = Spectrogram.MelSpectrogram(n_fft=n_fft, n_mels=n_mels, hop_length=512, verbose=False, device=device)
                start = time.time()
                for i in tqdm.tqdm(dataset):
                    i = i.to(device)
                    layer(i)
                result[f'Mel-{n_fft}-n_bins{n_mels}'] = time.time()-start
            else:
                continue

    # CQT

    for r in range(1,11):
        layer = Spectrogram.CQT1992v2(sr=44100, n_bins=84*r, bins_per_octave=12*r,hop_length=512,verbose=False,device=device)
        start = time.time()
        for i in tqdm.tqdm(dataset):
            i = i.to(device)
            layer(i)
        result[f'CQT-r={r}'] = time.time()-start

    pickle.dump(result, open(Path(__file__).parent /'./Pytorch_result', 'wb'))