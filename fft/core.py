import numpy as np


def DFT_slow(x):
    """
    Computes the discrete Fourier Transform of the 1D array x.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def FFT(x):
    """
    A recursive implementation of the 1D Cooley-Tukey FFT.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("Size of x must be a power of two...")
    elif N <= 32:
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate(
            [X_even + factor[: N // 2] * X_odd, X_even + factor[N // 2 :] * X_odd]
        )


def FFT_vectorized(x):
    """
    A vectorized, non-recursive version of the Cooley-Tukey FFT.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("Size of x must be a power of two...")

    N_min = min(N, 32)

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    while X.shape[0] < N:
        X_even = X[:, : X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2 :]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])

    return X.ravel()


if __name__ == "__main__":
    x = np.random.random(1024)
    print(np.allclose(DFT_slow(x), np.fft.fft(x)))
    print(np.allclose(FFT(x), np.fft.fft(x)))
    print(np.allclose(FFT_vectorized(x), np.fft.fft(x)))
