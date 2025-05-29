# CUDA Programming in Python for Signal Processing - Beginner Guide

from numba import cuda
import numpy as np
import cupy as cp
import time


def timeit_DND(func_DND):
    def wrapper_DND():
        t_start_DND = time.time()
        func_DND()
        t_end_DND = time.time()
        print(t_end_DND - t_start_DND)
    return wrapper_DND

# ------------------ CUDA BASICS ------------------

# In CUDA, the GPU runs many small programs in parallel.
# These programs are called "threads".
# Threads are grouped into "blocks".
# Blocks are grouped into a "grid".

# We use thread/block/grid to break a big task into smaller parallel pieces.

# ------------------ Decorators ------------------

# @cuda.jit is a decorator provided by Numba to define a CUDA kernel.

# A CUDA kernel is a special function that runs on the GPU.
# It is executed in parallel by many threads.

# When you use @cuda.jit without arguments, it creates a device function or
# a CPU fallback version.
# When you use it with arguments like @cuda.jit(...), it creates a GPU kernel.


# ------------------ Transfer of data ------------------

# test = [1,2,3,4,5,6,7,8,9,10] # Array is in memory (RAM)

# test_GPU = cp.asarray(test) #copy array from RAM into GPU memory

# test_RAM = test_GPU.get() # copy array from GPU Memory to RAM





# CUDA kernel to add two arrays
@cuda.jit
def add_arrays(a, b, result):
    
    # Get the global thread index
    idx = cuda.grid(1)
    # Make sure we don't go out of bounds
    if idx < a.size:
        result[idx] = a[idx] + b[idx]


def overlap_add_DND(x,h):
    N = 2**len(h) # BLOCK_SIZE
    M = len(h)
    L = N - M + 1
    h = h + [0 for _ in range(L - 1)]
    x_ = [[0] * (len(x)/L)] * (L+M-1) # matrix = [[..]*row]*col
    for i in range(len(x)):
        for j in range(len(x)/L):
            x_[i%(len(x)/L),j] = x[i]
            

    """if idx < len(x)/L:
        x_[0:L,idx] = x[idx*L:(idx+1)*L]"""
    




# Size of arrays
N = 3

# Initialize arrays on the host (CPU)
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
result = np.zeros(N, dtype=np.float32)

# Copy arrays to the device (GPU)
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_result = cuda.device_array_like(result)


# # Set up CUDA configuration
# threads_per_block = 128
# blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# # Launch the kernel
# add_arrays[blocks_per_grid, threads_per_block](d_a, d_b, d_result)

# # Copy the result back to the host
# result = d_result.copy_to_host()

# At this point, result = a + b (element-wise addition)
