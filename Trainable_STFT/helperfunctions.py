from tqdm import tqdm
import numpy as np

def get_dummy_dataset():
    fs = 44100
    interval = 0.2
    t = np.linspace(0, interval, int(fs*interval))
    fmin = 200

    X = np.zeros(((fs//2-fmin)//2*10, int(fs*interval)), np.float32)
    Y = np.zeros((fs//2-fmin)//2*10, np.float32)
    counter=0
    for k in tqdm(range(fmin,fs//2, 2)): # Creating signals with different frequencies
         for phi in np.arange(0,1,0.1):
            X[counter]=np.sin(2*np.pi*(k*t+phi))
            Y[counter]=k/interval
            counter+=1
    return X, Y