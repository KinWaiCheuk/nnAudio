#!/bin/bash

# Part 1 MAPS dataset
# Running the experiments in GPU
python ./MAPS/STFT.py kapre
python ./MAPS/Melspectrogram.py kapre
python ./MAPS/STFT.py torchaudio
python ./MAPS/Melspectrogram.py torchaudio