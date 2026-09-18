#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // c[i] = a[i] + b[i], guarded by i < n
       int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        c[i]=a[i]+b[i];
    }
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    // allocate a, b, c on the device; copy a and b over (Host -> Device);
    // launch the kernel; copy c back; free memory
    float *d_a;
    float*d_b;
    float* d_c;
    int n=a.size();
    cudaMalloc(&d_a,a.size()*sizeof(float));
    cudaMalloc(&d_b,b.size()*sizeof(float));
    cudaMalloc(&d_c,a.size()*sizeof(float));
    cudaMemcpy(d_a,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice);
    int blockSize = 256;
int numblocks=(n+blockSize-1)/blockSize;
//int blockSize = 256;
    add_kernel<<<numblocks,blockSize>>>(d_a,d_b,d_c,n);
cudaDeviceSynchronize();//device=gpu
std::vector<float>result(n);
cudaMemcpy(result.data(), d_c, n*sizeof(float) , cudaMemcpyDeviceToHost);
return result;

    return {};
}

//a.data()=returns a pointer to the first element of the vector.
