import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

div255_mod = SourceModule("""
    __global__ void div255(float *cal_array)
    {
      int idx = blockDim.x*blockIdx.x + threadIdx.x;
      cal_array[idx] /= 255.0;
    }
    """)
