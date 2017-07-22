import numpy as np
import math

def corr_master(corrtype, cutoff, *args):

    nargin = len(args)
    if corrtype == 'auto':
        H1 = args[0]
        H1 = np.fft.fftn(H1)
        H1 = H1*np.conj(H1)
        H1 = np.fft.fftshift(np.fft.ifftn(H1))
        GG = H1
        if H1.ndim == 2:
            ggx = math.floor(GG.shape[0]/2+1)
            ggy = math.floor(GG.shape[1]/2+1)
            GG = GG[(ggx-cutoff):(ggx+cutoff), (ggy-cutoff):(ggy+cutoff)]
        elif H1.ndim ==3:
            ggx = math.floor(GG.shape[0]/2+1)
            ggy = math.floor(GG.shape[1]/2+1)
            ggz = math.floor(GG.shape[2]/2+1)
            GG = GG[(ggx-cutoff):(ggx+cutoff), (ggy-cutoff):(ggy+cutoff), (ggz-cutoff):(ggz+cutoff)]
        else:
            print('Incorrect Number of Dimensions!')
        return GG
    elif corrtype == 'cross':
        H1 = args[0]
        H2 = args[1]
        H1 = np.fft.fftn(H1)
        H1 = np.conj(H1)
        H2 = np.fft.fftn(H2)
        H1 = H1*H2
        H1 = np.fft.fftshift(np.fft.ifftn(H1))
        GG = H1
        if H1.ndim == 2:
            ggx = math.floor(GG.shape[0]/2+1)
            ggy = math.floor(GG.shape[1]/2+1)
            GG = GG[(ggx-cutoff):(ggx+cutoff), (ggy-cutoff):(ggy+cutoff)]
        elif H1.ndim ==3:
            ggx = math.floor(GG.shape[0]/2+1)
            ggy = math.floor(GG.shape[1]/2+1)
            ggz = math.floor(GG.shape[2]/2+1)
            GG = GG[(ggx-cutoff):(ggx+cutoff), (ggy-cutoff):(ggy+cutoff), (ggz-cutoff):(ggz+cutoff)]
        else:
            print('Incorrect Number of Dimensions!')
        return GG
    else:
        print("Please either <Auto> or <Cross> as correlation type")
