import numpy as np

from .core import DFT_slow, FFT, FFT_vectorized


def test_DFT_slow():
    x = np.random.random(1024)
    assert np.allclose(DFT_slow(x), np.fft.fft(x))


def test_FFT():
    x = np.random.random(1024)
    assert np.allclose(FFT(x), np.fft.fft(x))


def test_FFT_vectorized():
    x = np.random.random(1024)
    assert np.allclose(FFT_vectorized(x), np.fft.fft(x))
