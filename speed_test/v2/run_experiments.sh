#!/bin/bash

# Part 1 MAPS dataset
# Running the experiments in GPU
python ./MAPS/STFT.py GPU
python ./MAPS/Melspectrogram.py GPU
python ./MAPS/CQT1992.py GPU
python ./MAPS/CQT2010.py GPU
python ./MAPS/CQT2010v1.py GPU

# Running the experiments in CPU
python ./MAPS/STFT.py CPU
python ./MAPS/Melspectrogram.py CPU
python ./MAPS/CQT1992.py CPU
python ./MAPS/CQT2010.py CPU
python ./MAPS/CQT2010v1.py CPU

# Running the experiments with librosa
python ./MAPS/STFT.py librosa
python ./MAPS/Melspectrogram.py librosa
python ./MAPS/CQT2010.py librosa


# Part 2 MusicNet
python ./MusicNet/Pytorch_speedtest.py
python ./MusicNet/librosa_speedtest.py