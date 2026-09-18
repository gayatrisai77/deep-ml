#include <cuda_runtime.h>
#include <vector>

__global__ void scale_kernel(const float* x, float a, float* y, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        y[i]=a*x[i];
    }
    // y[i] = a * x[i]
}

std::vector<float> scalar_multiply(const std::vector<float>& x, float a) {
    int n=x.size();
   std:: vector<float>result(n);
    float* d_input;
    cudaMalloc(&d_input, n*sizeof(float));//=x
    float* d_out;
    cudaMalloc(&d_out, n*sizeof(float));//=y
    cudaMemcpy(d_input,x.data(),n*sizeof(float),cudaMemcpyHostToDevice);
    int noofblocks=(n+256-1)/256;
scale_kernel<<<noofblocks,256>>>(d_input,a,d_out,n);
    cudaDeviceSynchronize();//gpu synchronise after the gpu call
    cudaMemcpy(result.data(),d_out,n*sizeof(float),cudaMemcpyDeviceToHost);
    return result;
    return {};
}
