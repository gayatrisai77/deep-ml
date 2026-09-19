#include <cuda_runtime.h>
#include <vector>

__global__ void square_kernel(const float* x, float* out, int n) {
    // Use a grid-stride loop:
    //   for (int i = start; i < n; i += stride) out[i] = x[i] * x[i];
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    int start=i;
    int stride=blockDim.x*gridDim.x;
    //gridDim.x means the number of grids or blocks
    for(int j=start;j<n;j+=stride){
        out[j]=x[j]*x[j];
    }
}

std::vector<float> square(const std::vector<float>& x) {
 
    float* d_out;
    float* d_x;
    int n=x.size();
    cudaMalloc(&d_x,n*sizeof(float));
    cudaMalloc(&d_out,n*sizeof(float));//here just the output gets stored
    cudaMemcpy(d_x, x.data(), n*sizeof(float), cudaMemcpyHostToDevice);//cpu to gpu
    int blockSize = 256;
int numBlocks = (n+256-1)/256;

square_kernel<<<numBlocks, blockSize>>>(d_x, d_out, n);
 //   square_kernel<<<>>>(d_x,d_out,n);
    cudaDeviceSynchronize();//device=gpu 
   std::vector<float>result(n);
   cudaMemcpy(result.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost);//gpu to cpu
   return result;
    return {};
}
