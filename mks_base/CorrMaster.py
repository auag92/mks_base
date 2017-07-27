import numpy as np

from toolz.curried import curry
from toolz.curried import map
from toolz.curried import pipe


fft = curry(np.fft.fft)  # pylint: disable=invalid-name

ifft = curry(np.fft.ifft)  # pylint: disable=invalid-name

fftn = curry(np.fft.fftn)  # pylint: disable=invalid-name

ifftn = curry(np.fft.ifftn)  # pylint: disable=invalid-name

fftshift = curry(np.fft.fftshift)  # pylint: disable=invalid-name

conj = curry(np.conj)

func = curry(lambda x, y: conj(x) * fftn(y))

@curry
def return_slice(x_data, cutoff):

    s = np.asarray(x_data.shape).astype(int) // 2 + 1

    if x_data.ndim == 2:
        return x_data[(s[0] - cutoff):(s[0] + cutoff),
                      (s[1] - cutoff):(s[1] + cutoff)]
    elif x_data.ndim ==3:
        return x_data[(s[0] - cutoff):(s[0] + cutoff),
                      (s[1] - cutoff):(s[1] + cutoff),
                      (s[2] - cutoff):(s[2] + cutoff)]
    else:
        print('Incorrect Number of Dimensions!')


@curry
def corr_master_(x_data, cutoff, func):
    return pipe(x_data,
                fftn,
                func,
                ifftn,
                fftshift,
                return_slice(cutoff=cutoff))


@curry
def corr_master_auto(cutoff, *args):
    """
    Returns auto-corrlation of and input field with itself.
    """

    return pipe(args[0],
                fftn,
                lambda x: conj(x)*x,
                ifftn,
                fftshift,
                return_slice(cutoff=cutoff))


@curry
def corr_master_cross(cutoff, *args):
    """
    Returns cross corrlation between two input fields.
    """
    return pipe(args[0],
                fftn,
                lambda x: conj(x) * fftn(args[1]),
                ifftn,
                fftshift,
                return_slice(cutoff = cutoff))


# @curry
# def corr_master_cross(cutoff, *args):
#     return pipe(args[0],
#                corr_master_(cutoff=cutoff, func=func(y=args[1])))


@curry
def corr_master(corrtype, cutoff, *args):
    """
    Wrapper function that returns auto or crosscorrelations for
    input fields by calling appropriate modules.
    """
    if corrtype == 'auto':
        return corr_master_auto(cutoff, *args)
    elif corrtype == 'cross':
        return corr_master_cross(cutoff, *args)
    else:
        print("Please either <Auto> or <Cross> as correlation type")
