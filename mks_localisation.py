import numpy as np
import numpy.fft as fft
from scipy.linalg import lstsq
from sklearn.linear_model import LinearRegression

    
def axes_(arr):
    return (np.arange(arr.ndim - 1)+1)


def axes_shape(arr):
    return arr.shape


def fit(X, y, domain, n_states, discretize):
    
    X_ = discretize(X, domain, n_states)
    _axes=axes_(X)
    FX = fft.fftn(X_, axes=_axes)
    Fy = fft.fftn(y,  axes=_axes)
    
    Fkernel = np.zeros(FX.shape[1:], dtype=np.complex)
    lstsq_rcond = np.finfo(float).eps*1e4
    s0 = (slice(None),)
    for ijk in np.ndindex(FX.shape[1:-1]):
        s1 = s0
        Fkernel[ijk + s1] = lstsq(FX[s0 + ijk + s1], Fy[s0 + ijk],
                                  lstsq_rcond)[0]
    return Fkernel


def predict(X, coeff, domain, n_states, discretize): 
    
    X_  = discretize(X, domain, n_states)
    FX  = fft.fftn(X_, axes=axes_(X))
    Fy  = np.sum(FX*coeff[None], axis = -1)
    y   = fft.ifftn(Fy, axes=axes_(X)).real
    return y


def coeff_real(coeff, _axes):
    return fft.ifftshift(fft.ifftn(coeff, axes = _axes)).real

