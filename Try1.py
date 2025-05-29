# Simple CUDA Overlap-Add and Overlap-Save Example in Python
import numpy as np
from numba import cuda
import cupy as cp

def overlap_add_gpu(x, h, block_size):
    x = cp.asarray(x)
    h = cp.asarray(h)
    M = h.size
    L = block_size
    N = L + M - 1
    H = cp.fft.fft(h, N)
    y = cp.zeros(x.size + M - 1, dtype=cp.float32)
    for i in range(0, x.size, L):
        x_block = x[i:i+L]
        if x_block.size < L:
            x_block = cp.pad(x_block, (0, L - x_block.size))
        X = cp.fft.fft(x_block, N)
        Y = X * H
        y_block = cp.fft.ifft(Y).real
        y[i:i+N] += y_block
    return cp.asnumpy(y)

def overlap_save_gpu(x, h, block_size):
    x = cp.asarray(x)
    h = cp.asarray(h)
    M = h.size
    L = block_size
    N = L + M - 1
    H = cp.fft.fft(h, N)
    x_padded = cp.pad(x, (M-1, 0))
    y = []
    for i in range(0, x.size, L):
        x_block = x_padded[i:i+N]
        if x_block.size < N:
            x_block = cp.pad(x_block, (0, N - x_block.size))
        X = cp.fft.fft(x_block, N)
        Y = X * H
        y_block = cp.fft.ifft(Y).real
        y.append(y_block[M-1:])
    y = cp.concatenate(y)[:x.size + M - 1]
    return cp.asnumpy(y)

# Example usage
if __name__ == "__main__":
    # Create random signal and filter
    x = np.random.rand(10000).astype(np.float32)
    h = np.random.rand(256).astype(np.float32)
    block_size = 1024

    # Choose method: 'ola' for overlap-add, 'ols' for overlap-save
    method = 'ola'  # Change to 'ols' for overlap-save

    if method == 'ola':
        y = overlap_add_gpu(x, h, block_size)
        print("Overlap-Add result (first 10 samples):", y[:10])
    else:
        y = overlap_save_gpu(x, h, block_size)
        print("Overlap-Save result (first 10 samples):", y[:10])




import numpy as np
import matplotlib.pyplot as plt

def overlap_add(x, h, block_size):
    """Perform convolution using the overlap-add method."""
    L = len(h)
    N = block_size + L - 1
    H = np.fft.rfft(h, n=N)
    
    # Pad input to a multiple of block_size
    padded_length = int(np.ceil(len(x) / block_size) * block_size)
    x_padded = np.pad(x, (0, padded_length - len(x)))
    
    # Allocate output
    y = np.zeros(len(x_padded) + L - 1)
    
    # Process each block
    for i in range(0, padded_length, block_size):
        block = x_padded[i:i + block_size]
        X = np.fft.rfft(block, n=N)
        Y = X * H
        y_block = np.fft.irfft(Y, n=N)
        y[i:i + N] += y_block
    
    # Trim to original convolution length
    return y[:len(x) + len(h) - 1]

# Example signals
np.random.seed(0)
x = np.random.randn(500)  # input signal
h = np.array([0.2, 0.5, 0.3, -0.1])  # FIR filter
block_size = 128

# Compute convolution
y_overlap_add = overlap_add(x, h, block_size)
y_direct = np.convolve(x, h)

# Plot results
plt.figure()
plt.plot(y_direct, label='Direct Convolution')
plt.plot(y_overlap_add, label='Overlap-Add', linestyle='--')
plt.title('Overlap-Add vs Direct Convolution')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
