import numpy as np
import librosa

#LPC spectrum generator vectorized
def LPCSpect_Fast(signal,sr,order,freqs,window='hanning'):
    '''
    Function to compute the LPC spectrum of a signal at specified set of frequencies
    signal = the (typically short) section of signal we want to calculate the spectrum of 
    sr = sample rate of signal
    order = order of lpc polynomial
    freqs = vector frequencies at which spectrum will be evaluated
    window = window function to apply to signal
    '''
    #Make sure signal is numpy array
    signal = np.array(signal)
    #Signal length
    N=signal.shape[0]
    #Sampling interval (seconds)
    delta = 1.0/sr
    #Set up window function
    if window == 'hanning':
        win = np.hanning(N+1)[:-1]
    else:
        win = np.ones(N)
    
    x = signal*win
    a = librosa.lpc(x, order=order)
    
    delta = 1.0/sr
    
    j = complex(0,1)
    ks = np.arange(order+1)
    zs = np.exp(2*np.pi*j*delta*np.outer(freqs,ks)) 
    
    Pxx = 0.5*np.log10(np.abs(x).sum()) + np.log10(1/np.abs(zs@a)**2)
    return Pxx