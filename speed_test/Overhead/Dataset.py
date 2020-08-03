# For the dataloading
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from abc import abstractmethod
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
import soundfile
import numpy as np
import os
from constants import *

class PianoRollAudioDataset_nnAudio(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group): #self.files is defined in MAPS class
                self.data.append(self.load(*input_files)) # self.load is a function defined below. It first loads all data into memory first
    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])

        audio_length = len(data['audio'])
        step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
        n_steps = self.sequence_length // HOP_LENGTH
        step_end = step_begin + n_steps

        begin = step_begin * HOP_LENGTH
        end = begin + self.sequence_length

        result['audio'] = data['audio'][begin:end]
        result['label'] = data['label'][step_begin:step_end, :]
        result['velocity'] = data['velocity'][step_begin:step_end, :]


        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15
        result['frame'] = (result['label'] > 1).float()
        # print(f"result['audio'].shape = {result['audio'].shape}")
        # print(f"result['label'].shape = {result['label'].shape}")
        return result

    def __len__(self):
        return len(self.data)

    @classmethod # This one seems optional?
    @abstractmethod # This is to make sure other subclasses also contain this method
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels
        Returns
        -------
            A dictionary containing the following data:
            path: str
                the path to the audio file
            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform
            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path) and self.refresh==False: # Check if .pt files exist, if so just load the files
            return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio, sr = soundfile.read(audio_path, dtype='int16')
#         audio, sr = wavfile.read(audio_path)
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio) # convert numpy array to pytorch tensor
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1 # This will affect the labels time steps

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
            onset_right = min(n_steps, left + HOPS_IN_ONSET) # Ensure the time step of onset would not exceed the last time step
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
#         torch.save(data, saved_data_path)
        return data

class MusicNet_nnAudio(PianoRollAudioDataset_nnAudio):
    def __init__(self, path='../IJCNN2020_music_transcription/data/', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'test']

    def files(self, group):

        wavs = sorted(glob(os.path.join(self.path, f'{group}_data/*.wav')))
        tsvs = sorted(glob(os.path.join(self.path, f'tsv_{group}_labels/*.tsv')))
        assert(all(os.path.isfile(wav) for wav in wavs))

        return zip(wavs, tsvs)
    
class PianoRollAudioDataset_librosa(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, refresh=False, save_name='librosa', device='cpu'):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh
        self.save_name = save_name
        print(f'save_name1 = {self.save_name}')

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group): #self.files is defined in MAPS class
                self.data.append(self.load(*input_files)) # self.load is a function defined below. It first loads all data into memory first
    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])

        spec_length = data['audio'].size(-1)
        n_steps = self.sequence_length // HOP_LENGTH   
        step_begin = self.random.randint(spec_length - n_steps)
        
        step_end = step_begin + n_steps

        begin = step_begin * HOP_LENGTH

        result['audio'] = data['audio'][:,step_begin:step_end]
        result['label'] = data['label'][step_begin:step_end, :]
        result['velocity'] = data['velocity'][step_begin:step_end, :]


        result['audio'] = result['audio'].float() # converting to float by dividing it by 2^15
        result['frame'] = (result['label'] > 1).float()
        # print(f"result['audio'].shape = {result['audio'].shape}")
        # print(f"result['label'].shape = {result['label'].shape}")
     
        
        return result

    def __len__(self):
        return len(self.data)

    @classmethod # This one seems optional?
    @abstractmethod # This is to make sure other subclasses also contain this method
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels
        Returns
        -------
            A dictionary containing the following data:
            path: str
                the path to the audio file
            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform
            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '_'+self.save_name)
        if os.path.exists(saved_data_path) and self.refresh==False: # Check if .pt files exist, if so just load the files
            return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio = np.load(audio_path)
#         audio, sr = wavfile.read(audio_path)

        audio = torch.FloatTensor(audio) # convert numpy array to pytorch tensor
        spec_length = audio.size(-1)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = spec_length # This will affect the labels time steps

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
            onset_right = min(n_steps, left + HOPS_IN_ONSET) # Ensure the time step of onset would not exceed the last time step
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        if self.save_name != None:    
            torch.save(data, saved_data_path[:-3]+'_'+self.save_name)
        else:
            pass
        return data

class MusicNet_Spec(PianoRollAudioDataset_librosa):
    def __init__(self, path='../IJCNN2020_music_transcription/data/', groups=None, sequence_length=None, seed=42, refresh=False, save_name='librosa', device='cpu'):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, refresh, save_name, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'test']

    def files(self, group):

        wavs = sorted(glob(os.path.join(self.path, f'{group}_data/*.spec.npy')))
        tsvs = sorted(glob(os.path.join(self.path, f'tsv_{group}_labels/*.tsv')))
        assert(all(os.path.isfile(wav) for wav in wavs))

        return zip(wavs, tsvs)