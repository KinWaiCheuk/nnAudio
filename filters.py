from __future__ import print_function
from subprocess import call
import torch.utils.data as data
import os,mmap
import os.path
import pickle
import errno
import csv
import numpy as np
import torch
import pandas as pd
from time import time
from tqdm import tqdm
import math
import mir_eval
from sklearn.metrics import average_precision_score

from intervaltree import IntervalTree
from scipy.io import wavfile

sz_float = 4    # size of a float
epsilon = 10e-8 # fudge factor for normalization

class MusicNet(data.Dataset):
    """`MusicNet <http://homes.cs.washington.edu/~thickstn/musicnet.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``train_data``,
            otherwise from ``test_data``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        mmap (bool, optional): If true, mmap the dataset for faster access times.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        window (int, optional): Size in samples of a data point.
        pitch_shift (int,optional): Integral pitch-shifting transformations.
        jitter (int, optional): Continuous pitch-jitter transformations.
        epoch_size (int, optional): Designated Number of samples for an "epoch"
    """
    url = 'https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz'
    raw_folder = 'raw'
    train_data, train_labels, train_tree = 'train_data', 'train_labels', 'train_tree.pckl'
    test_data, test_labels, test_tree = 'test_data', 'test_labels', 'test_tree.pckl'
    extracted_folders = [train_data,train_labels,test_data,test_labels]

    def __init__(self, root, train=True, download=False, refresh_cache=False, mmap=True, normalize=True, window=16384,sequence=None, pitch_shift=0, jitter=0., epoch_size=100000):
        self.refresh_cache = refresh_cache
        self.mmap = mmap
        self.normalize = normalize
        self.window = window
        self.pitch_shift = pitch_shift
        self.jitter = jitter
        self.size = epoch_size
        self.m = 128
        self.train = train
        self.sequence = sequence
#         self.counter = 0

        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if train:
            self.data_path = os.path.join(self.root, self.train_data)
            labels_path = os.path.join(self.root, self.train_labels, self.train_tree)
        else:
            self.data_path = os.path.join(self.root, self.test_data)
            labels_path = os.path.join(self.root, self.test_labels, self.test_tree)

        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

        self.rec_ids = list(self.labels.keys())
        self.records = dict()
        self.open_files = []

    def __enter__(self):
        for record in os.listdir(self.data_path):
            if not record.endswith('.bin'): continue
            if self.mmap:
                fd = os.open(os.path.join(self.data_path, record), os.O_RDONLY)
                buff = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                self.records[int(record[:-4])] = (buff, len(buff)/sz_float)
                self.open_files.append(fd)
            else:
                f = open(os.path.join(self.data_path, record))
                self.records[int(record[:-4])] = (os.path.join(self.data_path, record),os.fstat(f.fileno()).st_size/sz_float)
                f.close()

    def __exit__(self, *args):
        if self.mmap:
            for mm in self.records.values():
                mm[0].close()
            for fd in self.open_files:
                os.close(fd)
            self.records = dict()
            self.open_files = []

    def access(self,rec_id,s,shift=0,jitter=0):
        """
        Args:
            rec_id (int): MusicNet id of the requested recording
            s (int): Position of the requested data point
            shift (int, optional): Integral pitch-shift data transformation
            jitter (float, optional): Continuous pitch-jitter data transformation
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        scale = 2.**((shift+jitter)/12.)

        if self.mmap:
            x = np.frombuffer(self.records[rec_id][0][s*sz_float:int(s+scale*self.window)*sz_float], dtype=np.float32).copy()
        else:
            fid,_ = self.records[rec_id]
#             start = time()
            with open(fid, 'rb') as f:
                f.seek(s*sz_float, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=int(scale*self.window))
#             x = torch.load(fid[:-4])
#             x = x[s:s+self.window]
#             print(time()-start)

        if self.normalize: x /= np.linalg.norm(x) + epsilon

        xp = np.arange(self.window,dtype=np.float32)
        x = np.interp(scale*xp,np.arange(len(x),dtype=np.float32),x).astype(np.float32)

        y = np.zeros(self.m,dtype=np.float32)
        for label in self.labels[rec_id][s+scale*self.window/2]:
            y[label.data[1]+shift] = 1

        return x,y

    def accessv2(self,rec_id,s,sequence,shift=0,jitter=0):
        """
        Args:
            rec_id (int): MusicNet id of the requested recording
            s (int): Position of the requested data point
            shift (int, optional): Integral pitch-shift data transformation
            jitter (float, optional): Continuous pitch-jitter data transformation
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """
        if sequence==None:
            sequence=1

        scale = 2.**((shift+jitter)/12.)

        if self.mmap:
            x = np.frombuffer(self.records[rec_id][0][s*sz_float:int(s+scale*self.window*sequence)*sz_float], dtype=np.float32).copy()
        else:
            fid,_ = self.records[rec_id]
#             start = time()
            with open(fid, 'rb') as f:
                f.seek(s*sz_float, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=int(scale*self.window*sequence))
#             x = torch.load(fid[:-4])
#             x = x[s:s+self.window]
#             print(time()-start)

        if self.normalize: x /= np.linalg.norm(x) + epsilon

        xp = np.arange(self.window*sequence,dtype=np.float32)
        x = np.interp(scale*xp,np.arange(len(x),dtype=np.float32),x).astype(np.float32)

        if sequence==1:
            y = np.zeros(self.m,dtype=np.float32)
            for label in self.labels[rec_id][s+scale*self.window/2]:
                y[label.data[1]+shift] = 1
        else: 
            y = np.zeros((sequence,self.m),dtype=np.float32)
            for num_frame in range(sequence):
                for label in self.labels[rec_id][(s + num_frame*self.window)+scale*self.window/2]:
                    y[num_frame,label.data[1]+shift] = 1

        return x,y

    def access_full(self,rec_id):
        """
        Args:
            rec_id (int): MusicNet id of the requested recording
        Returns:
            tuple: (audio, label)
        """

        fid,_ = self.records[rec_id]
#             start = time()
        with open(fid, 'rb') as f:
#                 f.seek(s*sz_float, os.SEEK_SET)
            x = np.fromfile(f, dtype=np.float32)
#             x = torch.load(fid[:-4])
#             x = x[s:s+self.window]
#             print(time()-start)

#         if self.normalize: x /= np.linalg.norm(x) + epsilon


        y = self.labels[rec_id]

        return x,y

    def __getitem__(self, index):
        """
        Args:
            index (int): (ignored by this dataset; a random data point is returned)
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """
        shift = 0
        if self.pitch_shift> 0:
            shift = np.random.randint(-self.pitch_shift,self.pitch_shift)

        jitter = 0.
        if self.jitter > 0:
            jitter = np.random.uniform(-self.jitter,self.jitter)

        rec_id = self.rec_ids[np.random.randint(0,len(self.rec_ids))]
        s = np.random.randint(0,self.records[rec_id][1]-(2.**((shift+jitter)/12.))*self.window)
#         return self.access(rec_id,s,shift,jitter)
        return self.accessv2(rec_id,s,self.sequence,shift,jitter)

    def __len__(self):
        return self.size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_data)) and \
            os.path.exists(os.path.join(self.root, self.test_data)) and \
            os.path.exists(os.path.join(self.root, self.train_labels, self.train_tree)) and \
            os.path.exists(os.path.join(self.root, self.test_labels, self.test_tree)) and \
            not self.refresh_cache

    def download(self):
        """Download the MusicNet data if it doesn't exist in ``raw_folder`` already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path):
            print('Downloading ' + self.url)
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                # stream the download to disk (it might not fit in memory!)
                while True:
                    chunk = data.read(16*1024)
                    if not chunk:
                        break
                    f.write(chunk)

        if not all(map(lambda f: os.path.exists(os.path.join(self.root, f)), self.extracted_folders)):
            print('Extracting ' + filename)
            if call(["tar", "-xf", file_path, '-C', self.root, '--strip', '1']) != 0:
                raise OSError("Failed tarball extraction")

        # process and save as torch files
        print('Processing...')

        if self.train:
            print('processing train data...')
            self.process_data(self.train_data)
            print('processing train labels...')
            trees = self.process_labels(self.train_labels)
            with open(os.path.join(self.root, self.train_labels, self.train_tree), 'wb') as f:
                pickle.dump(trees, f)
        else:
            print('processing test data...')
            self.process_data(self.test_data)
            print('processing test labels...')
            trees = self.process_labels(self.test_labels)
            with open(os.path.join(self.root, self.test_labels, self.test_tree), 'wb') as f:
                pickle.dump(trees, f)



        self.refresh_cache = False
        print('Download Complete')

    # write out wavfiles as arrays for direct mmap access
    def process_data(self, path):
        for item in tqdm(os.listdir(os.path.join(self.root,path))):
            if not item.endswith('.wav'): continue
            uid = int(item[:-4])
            _, data = wavfile.read(os.path.join(self.root,path,item))
            data.tofile(os.path.join(self.root,path,item[:-4]+'.bin'))

    # wite out labels in intervaltrees for fast access
    def process_labels(self, path):
        trees = dict()
        for item in tqdm(os.listdir(os.path.join(self.root,path))):
            if not item.endswith('.csv'): continue
            uid = int(item[:-4])
            tree = IntervalTree()
#             df = pd.read_csv(os.path.join(self.root, path, item))

#             for label in df.iterrows():
#                 start_time = label[1]['start_time']
#                 end_time = label[1]['end_time']
#                 instrument = label[1]['instrument']
#                 note = label[1]['note']
#                 start_beat = label[1]['start_beat']
#                 end_beat = round(label[1]['end_beat'], 15) # Round to prevent float number precision problem
#                 note_value = label[1]['note_value']
#                 tree[start_time:end_time] = (instrument,note,start_beat,end_beat,note_value)

            with open(os.path.join(self.root,path,item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_time:end_time] = (instrument,note,start_beat,end_beat,note_value)
            trees[uid] = tree
        return trees

def create_fourier_kernals(n_fft, freq_bins=None, low=50,high=6000, sr=44100, freq_scale='linear', windowing="no"):
    if freq_bins==None:
        freq_bins = n_fft//2+1
    
    s = np.arange(0, n_fft, 1.)
    wsin = np.empty((freq_bins,1,n_fft))
    wcos = np.empty((freq_bins,1,n_fft))
    start_freq = low
    end_freq = high
    

    # num_cycles = start_freq*d/44000.
    # scaling_ind = np.log(end_freq/start_freq)/k
    
    if windowing=="no":
        window_mask = 1
    elif windowing=="hann":
        window_mask = 0.5-0.5*np.cos(2*np.pi*s/(n_fft)) # same as hann(n_fft, sym=False)
    else:
        raise Exception("Unknown windowing mode, please chooes either \"no\" or \"hann\"")
        
    if freq_scale == 'linear':
        start_bin = start_freq*n_fft/sr
        scaling_ind = (end_freq/start_freq)/freq_bins
        for k in range(freq_bins): # Only half of the bins contain useful info
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*(k*scaling_ind*start_bin)*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*(k*scaling_ind*start_bin)*s/n_fft)
    elif freq_scale == 'log':
        start_bin = start_freq*n_fft/sr
        scaling_ind = np.log(end_freq/start_freq)/freq_bins
        for k in range(freq_bins): # Only half of the bins contain useful info
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)   
    elif freq_scale == 'no':
        for k in range(freq_bins): # Only half of the bins contain useful info
            wsin[k,0,:] = window_mask*np.sin(2*np.pi*k*s/n_fft)
            wcos[k,0,:] = window_mask*np.cos(2*np.pi*k*s/n_fft)
    else:
        print("Please select the correct frequency scale, 'linear' or 'log'")
    
    return wsin,wcos

def get_mir_accuracy(Yhat, Y_true, threshold=0.4, m=128):
    if isinstance(Yhat, torch.Tensor):
        Yhat = Yhat.cpu().numpy()
    Yhatlist = []
    Ylist = []
    Yhatpred = Yhat>threshold
    print(f"Calculating accuracy ...", end = '\r')
    for i in range(len(Yhatpred)):
        fhat = []
        f = []
        for note in range(m):
            if Yhatpred[i][note] == 1:
                fhat.append(440.*2**(((note)-69.)/12.))

            if Y_true[i][note] == 1:
                f.append(440.*2**(((note)-69.)/12.))

        Yhatlist.append(np.array(fhat))
        Ylist.append(np.array(f))
    avp = average_precision_score(Y_true.flatten(),Yhat.flatten())
    P,R,Acc,Esub,Emiss,Efa,Etot,cP,cR,cAcc,cEsub,cEmiss,cEfa,cEtot = \
    mir_eval.multipitch.metrics(np.arange(len(Ylist))/100.,Ylist,np.arange(len(Yhatlist))/100.,Yhatlist)
    print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(100*avp,100*P,100*R,Acc,Etot,Esub,Emiss,Efa))
    return avp,P,R,Acc,Etot
def get_piano_roll(rec_id, test_set, model, device, window=16384, stride=1000, offset=44100, count=7500, batch_size=500, m=128):
    sf=4
    if stride == -1:
        stride = (test_set.records[rec_id][1] - offset - int(sf*window))/(count-1)
        stride = int(stride)
    else:
        count = (test_set.records[rec_id][1] - offset - int(sf*window))/stride + 1
        count = int(count)
        
    X = np.zeros([count, window])
    Y = np.zeros([count, m])    
        
    for i in range(count):
        X[i,:], Y[i] =  test_set.access(rec_id, offset+i*stride)
    
    with torch.no_grad():
        Y_pred = torch.zeros([count,m])
        for i in range(len(X)//batch_size):
            print(f"{i}/{(len(X)//batch_size)} batches", end = '\r')
            X_batch = torch.tensor(X[batch_size*i:batch_size*(i+1)]).float().to(device)
            Y_pred[batch_size*i:batch_size*(i+1)] = model(X_batch).cpu()
    
    return Y_pred, Y


# def get_mir_accuracy(Yhat, Y_true, threshold=0.4, m=128):
#     Yhatlist = []
#     Ylist = []
#     Yhatpred = Yhat>threshold
#     for i in range(len(Yhatpred)):
#         print(f"{i}/{len(Yhatpred)} batches", end = '\r')
#         fhat = []
#         f = []
#         for note in range(m):
#             if Yhatpred[i][note] == 1:
#                 fhat.append(440.*2**(((note)-69.)/12.))

#             if Y_true[i][note] == 1:
#                 f.append(440.*2**(((note)-69.)/12.))

#         Yhatlist.append(np.array(fhat))
#         Ylist.append(np.array(f))
#     avp = average_precision_score(Y_true.flatten(),Yhat.detach().cpu().flatten())
#     P,R,Acc,Esub,Emiss,Efa,Etot,cP,cR,cAcc,cEsub,cEmiss,cEfa,cEtot = \
#     mir_eval.multipitch.metrics(np.arange(len(Ylist))/100.,Ylist,np.arange(len(Yhatlist))/100.,Yhatlist)
#     print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(100*avp,100*P,100*R,Acc,Etot,Esub,Emiss,Efa))
#     return avp,P,R,Acc,Etot
# def get_piano_roll(rec_id, test_set, model, window=16384, stride=1000, offset=44100, count=7500, batch_size=500, m=128):
#     sf=4
#     if stride == -1:
#         stride = (test_set.records[rec_id][1] - offset - int(sf*window))/(count-1)
#         stride = int(stride)
#     else:
#         count = (test_set.records[rec_id][1] - offset - int(sf*window))/stride + 1
#         count = int(count)
        
#     X = np.zeros([count, window])
#     Y = np.zeros([count, m])    
        
#     for i in range(count):
#         X[i,:], Y[i] =  test_set.access(rec_id, offset+i*stride)
        
#     Y_pred = torch.zeros([count,m])
#     for i in range(len(X)//batch_size):
#         X_batch = torch.tensor(X[batch_size*i:batch_size*(i+1)]).float().cuda()   
#         Y_pred[batch_size*i:batch_size*(i+1)] = model(X_batch)
    
#     return Y_pred, Y
    