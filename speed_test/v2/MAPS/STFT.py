import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import argparse
from librosa import stft
import tqdm
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("device", type=str,help="Select device")
parser.add_argument("-d", "--GPU_device", type=str,help="Select GPU device")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.GPU_device

if __name__ == '__main__':
    if args.device=="CPU":
        device="cpu"
        print("using CPU")
    elif args.device in ["GPU", "torchaudio", 'tensorflow']:
        device=f"cuda:0"
        print("using GPU") 
    elif args.device=="librosa":
        print("using librosa")


    y_list = np.load(Path(__file__).parent /'./y_list.npy')

    if args.device in ["CPU", "GPU"]:
        import torch
        import torch.nn as nn
        from nnAudio import Spectrogram        
        y_torch = torch.tensor(y_list, device=device).float()
        spec_layer = Spectrogram.STFT(device=device)
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
        print('saving file')
        data.to_csv(Path(__file__).parent /f'./result/Spec_torch_{args.device}')

    elif args.device== "librosa":
        spec_list = []
        timing = []
        for e in range(5):
            t_start = time.time()
            for i in tqdm.tqdm(y_list, leave=True):
                spec = stft(i)
                spec_list.append(abs(spec))
            time_used = time.time()-t_start
            print(time_used)
            timing.append(time_used)

        print("mean = ",np.mean(timing))
        print("std = ", np.std(timing))


        data = pd.DataFrame(timing,columns=['t_avg'])
        data['Type'] = 'librosa'
        data.to_csv(Path(__file__).parent /f'./result/librosa_Spec')
        
    elif args.device== "kapre":
        import tensorflow as tf #tf 1.13.1
        from tensorflow.keras import Sequential
        import kapre.time_frequency as time_frequency
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction=0.3
        tf.Session(config=config)      
        spec_list = []
        model = Sequential()
        model.add(time_frequency.Spectrogram(n_dft=2048, n_hop=512, padding='same', input_shape=(1, 80000),
                                 power_spectrogram=2.0, return_decibel_spectrogram=False,
                                 trainable_kernel=False, image_data_format='default'))
        timing = []
        for e in range(20):
            t_start = time.time()
            spec = model.predict(y_list.reshape(1770,1,80000))
            time_used = time.time()-t_start
            print(time_used)
            timing.append(time_used)

        print("mean = ",np.mean(timing))
        print("std = ", np.std(timing))


        data = pd.DataFrame(timing,columns=['t_avg'])
        data['Type'] = 'kapre_GPU'
        data.to_csv(Path(__file__).parent /f'./result/kapre_Spec')
        
    elif args.device== "tensorflow":

        import tensorflow as tf
        input = tf.placeholder(tf.float32, shape=(None,80000))
        stft = tf.signal.stft(input, 2048, 512, pad_end=False)
        output_m = tf.abs(stft) 

        print('Extracting STFT using tf.signal...')
        timing = []
        with tf.Session() as sess:
            for e in range(20):
                t_start = time.time()
                sess.run(output_m, feed_dict={input:y_list[:1000]})
                sess.run(output_m, feed_dict={input:y_list[1000:]})
                time_used = time.time()-t_start
                print(time_used)
                timing.append(time_used)
        print('Test result for STFT using tf.signal')
        print("mean = ",np.mean(timing))
        print("std = ", np.std(timing))


        data = pd.DataFrame(timing,columns=['t_avg'])
        data['Type'] = 'tensorflow_GPU'
        data.to_csv(Path(__file__).parent /f'./result/Spec_tensorflow_GPU')           
        
        
    elif args.device== "torchaudio":
        import torchaudio
        import torch
        spec_list = []
        y_torch = torch.tensor(y_list, device=device).float()
        timing = []
        for e in range(20):
            t_start = time.time()
            specgram = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512).to(device)(y_torch)
            time_used = time.time()-t_start
            print(time_used)
            timing.append(time_used)

        print("mean = ",np.mean(timing))
        print("std = ", np.std(timing))


        data = pd.DataFrame(timing,columns=['t_avg'])
        data['Type'] = 'torchaudio_GPU'
        data.to_csv(Path(__file__).parent /f'./result/Spec_torchaudio_GPU')