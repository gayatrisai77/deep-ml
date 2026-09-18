#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        out[i]=i;
    }
}

std::vector<int> global_thread_indices(int n) {
    // 1. allocate device memory for n intsa
    // 2. launch the kernel with enough threads to cover n
    // 3. copy the result back to the host and return it
    int *d_out;
    cudaMalloc(&d_out, n*sizeof(int));
    int blocksize=256;
    int numblocks=(n+blocksize-1)/blocksize;
    index_kernel<<<numblocks,256>>>(d_out,n);
    cudaDeviceSynchronize();
    int* h_out=(int*)malloc(n*sizeof(int));//in the cpu/host 
    cudaMemcpy(h_out,d_out,n*sizeof(int),cudaMemcpyDeviceToHost);
    // Convert CPU array → C++ vector
    std::vector<int> result(h_out, h_out + n);
    return result;
    return {};
}
