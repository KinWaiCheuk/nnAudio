from librosa import stft, cqt
from librosa.feature import melspectrogram
import scipy
import os
import glob
import tqdm
import numpy as np
from time import sleep
import time
import pickle
from pathlib import Path
import os

p = os.path.join(Path(__file__).parent,'train_data', '*.wav')

if __name__ == '__main__':
    result = {}
    # STFT
    for n_fft in [256,512,1024,2048,4096]:
        print(f'n_fft={n_fft}')
        sleep(0.5)
        start = time.time()
        for file in tqdm.tqdm(glob.glob(p)):
            sr, wav = scipy.io.wavfile.read(file)
            output_name = os.path.basename(file)[:-4] + '_stft.sp'
            output = abs(stft(wav, n_fft=n_fft, hop_length=512))
            np.save('./dummy', output)
        result[f'librosa_STFT_{n_fft}'] = time.time()-start

    # Mel
    for n_fft in [512,1024,2048,4096]:
        for n_mels in [128, 256, 512, 1024, 2048]:
            if n_mels < n_fft:
                print(f'n_fft={n_fft}, n_mels={n_mels}')
                sleep(0.5)
                folder_dir = f'./4096mel-{n_mels}_output/'
                start = time.time()
                for file in tqdm.tqdm(glob.glob(p)):
                    sr, wav = scipy.io.wavfile.read(file)
                    output_name = os.path.basename(file)[:-4] + '_mel.sp'
                    output = melspectrogram(wav, sr=44100, n_fft=n_fft, n_mels=n_mels, hop_length=512)
                    np.save('./dummy', output)
                result[f'Mel-{n_fft}-n_bins{n_mels}'] = time.time()-start
            else:
                continue


    # CQT
    for r in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        folder_dir = f'./CQT-r={r}_output/'
        print(f'r = {r}')
        sleep(0.5)
        start = time.time()
        for file in tqdm.tqdm(glob.glob(p)):
            sr, wav = scipy.io.wavfile.read(file)
            output_name = os.path.basename(file)[:-4] + '_cqt.sp'
            output = abs(cqt(wav, n_bins=84*r, bins_per_octave=12*r))
            np.save('./dummy', output)
        result[f'CQT-r={r}'] = time.time()-start

    pickle.dump(result, open(Path(__file__).parent /'librosa_result', 'wb'))