import os, sys
import time

import torch
from torch.utils.cpp_extension import load_inline

import wurlitzer


os.environ['CXX'] = '/usr/lib/ccache/g++-11'
os.environ['CC'] = '/usr/lib/ccache/gcc-11'

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'


cuda_begin = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) {return (a + b - 1) / b;}
'''

torch_begin = r'''
#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
inline unsigned int cdiv(unsigned int a, unsigned int b) {return (a + b - 1) / b;}
'''


def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(
        cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
        extra_cuda_cflags=[flags], verbose=verbose, name=name
    )


def cdiv(a, b):
    "Int ceiling division of `a` over `b`"
    return (a + b - 1) // b


ext_name = 'test_ext'
cuda_path = ext_name + '.cu'
cuda_code = cuda_begin + r'''
__global__ void rgb_to_grayscale_kernel(unsigned char* out, unsigned char* in, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 0.2989f*in[i] + 0.5870f*in[i+n] + 0.1140f*in[i+2*n];
}

void launch_rgb_to_grayscale_kernel(unsigned int num_blocks, unsigned int num_threads, unsigned char* output, unsigned char* input, int n) {
    rgb_to_grayscale_kernel<<<num_blocks, num_threads>>>(output, input, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

cpp_path = ext_name + '.cpp'
cpp_code = torch_begin + r'''
void launch_rgb_to_grayscale_kernel(unsigned int num_blocks, unsigned int num_threads, unsigned char* output, unsigned char* input, int n);

torch::Tensor rgb_to_grayscale_out(torch::Tensor output, const torch::Tensor& input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    TORCH_CHECK((h == output.size(0)) || (w == output.size(1)) || (output.device() == input.device())
                || (output.scalar_type() == input.scalar_type()));
    int threads = 256;
    launch_rgb_to_grayscale_kernel(cdiv(w*h,threads), threads, output.data_ptr<unsigned char>(), input.data_ptr<unsigned char>(), w*h);
    return output;
}

torch::Tensor rgb_to_grayscale(const torch::Tensor& input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    auto output = torch::empty({h,w}, input.options());
    rgb_to_grayscale_out(output, input);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("rgb_to_grayscale", torch::wrap_pybind_function(rgb_to_grayscale), "rgb_to_grayscale");
m.def("rgb_to_grayscale_out", torch::wrap_pybind_function(rgb_to_grayscale_out), "rgb_to_grayscale_out");
}
'''

for p, c in [(cuda_path, cuda_code), (cpp_path, cpp_code)]:
    with open(p, 'w') as f: f.write(c)


def main():
    before_compile_t = time.time()
    module = torch.utils.cpp_extension.load(
        ext_name, [cuda_path, cpp_path],
        extra_cuda_cflags=['--ptxas-options=-v'],# -O3 -Xptxas -O3 -Xcompiler -O3'], # SM_89, compute_89
        verbose=True
    )
    print(f'compile took {time.time() - before_compile_t:.2f} secs')

    n = 2048
    t = torch.randint(0, 256, (3, n, n), dtype=torch.uint8, device="cuda")
    out = module.rgb_to_grayscale(t); torch.cuda.synchronize()

    t0 = time.perf_counter_ns()
    for i in range(10_000):
        module.rgb_to_grayscale_out(out, t)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()

    print((t1-t0) / 10_000 / 1_000, "µs") 

    with torch.profiler.profile() as prof:
        for i in range(10_000):
            module.rgb_to_grayscale_out(out, t)
            torch.cuda.synchronize()

    print(prof.key_averages().table())


if __name__ == '__main__':
    main()

