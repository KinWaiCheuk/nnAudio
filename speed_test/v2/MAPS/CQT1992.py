import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import pandas as pd
from nnAudio import Spectrogram
import os
import argparse
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser()
parser.add_argument("device", type=str,help="Select device")
args = parser.parse_args()


if args.device=="CPU":
    device="cpu"
    print("using CPU")
elif args.device=="GPU":
    device="cuda:0"
    print("using GPU")
elif args.device=="librosa":
    print("using librosa")

print(Path(__file__).parent /'./y_list.npy')
    
y_list = np.load(Path(__file__).parent /'./y_list.npy')

if args.device in ["CPU", "GPU"]:
    y_torch = torch.tensor(y_list, device=device).float()

    spec_layer = Spectrogram.CQT1992v2(sr=44100, n_bins=84, bins_per_octave=24, fmin=55, device=device)
    timing = []
    for e in range(20):
        t_start = time.time()
        spec = spec_layer(y_torch[:1000])
        spec = spec_layer(y_torch[1000:])
        time_used = time.time()-t_start
    #     print(time_used)
        timing.append(time_used)

    print("mean = ",np.mean(timing))
    print("std = ", np.std(timing))


    data = pd.DataFrame(timing,columns=['t_avg'])
    data['Type'] = f'torch_{args.device}'
    data.to_csv(Path(__file__).parent /f'./result/CQT1992_torch_{args.device}')

elif args.device== "librosa":
    print(f"There is no CQT1992 in librosa\nScript terminated...")
#     spec_list = []
#     timing = []
#     for e in range(5):
#         t_start = time.time()
#         for i in tqdm.tqdm(y_list, leave=True):
#             spec = stft(i)
#             spec_list.append(abs(spec))
#         time_used = time.time()-t_start
#         print(time_used)
#         timing.append(time_used)

#     print("mean = ",np.mean(timing))
#     print("std = ", np.std(timing))


#     data = pd.DataFrame(timing,columns=['t_avg'])
#     data['Type'] = 'librosa'
#     data.to_csv(f'./result/librosa_Spec')