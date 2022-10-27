# import pycuda module
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# import self custom pycuda module
from self_pycuda_mod import div255_mod

import numpy, time 
ARRAY_SIZE = 416*416*3

# sample data gen
a = numpy.random.randn(ARRAY_SIZE)
a = a.astype(numpy.float32)

# correct data used by numpy on CPU
time_sta = time.perf_counter() # time-cal
a_correct = numpy.divide(a, 255.)
cpu_numpy_time = time.perf_counter() - time_sta # time-cal

# molloc gpu mem
a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
cuda.memcpy_htod(a_gpu, a)# gpu mem -> cpu mem

# pycude exe
block   = (   96, 1, 1)      #  Block size 
grid    = ( 5408, 1, 1)      # Grid size
func = div255_mod.get_function("div255")
time_sta = time.perf_counter() # time-cal
func(a_gpu ,block=block , grid=grid, shared=0)
pycuda.autoinit.context.synchronize() # Wait for kernel completion before host access
gpu_cuda_time = time.perf_counter() - time_sta # time-cal

# gpu mem -> cpu mem
a_div255 = numpy.empty_like(a)
cuda.memcpy_dtoh(a_div255, a_gpu)

# diff numpy : gpu 
diff_log = [False if abs(a_correct[n] - a_div255[n])>0.0001 else True for n in range(ARRAY_SIZE)]
print("false") if  False in diff_log else print("success")# time-cal

print("time[s]\nnumpy-cpu : {:1.5f}\ncuda-gpu  : {:1.5f}".format(cpu_numpy_time, gpu_cuda_time))


## Memo
### Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
### Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)

### 416*416*3[pixel]を並列に計算したい...

