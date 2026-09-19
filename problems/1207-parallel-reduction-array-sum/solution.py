#include <cuda_runtime.h>
#include <vector>

__global__ void sum_kernel(const float* x, float* out, int n) {
    // 1. load x[tid] (or 0) into __shared__ memory, then __syncthreads()
    // 2. tree-reduce: for (s = blockDim.x/2; s > 0; s >>= 1) add sdata[tid+s]
    // 3. thread 0 writes out[0]
    __shared__ float sdata[256];
int tid=threadIdx.x;
if(tid<n){
    sdata[tid]=x[tid];
}//what if we have excess threads than the number of elements in the array
else{
    sdata[tid]=0.0f;
}
__syncthreads();
//so that values/data is placed correctly with sync!!! in sdata
for(int s=blockDim.x/2;s>0;s>>=1){
   if(tid<s){//these threads must only participate!
sdata[tid]+=sdata[tid+s];
   }
   __syncthreads();//we need to sync threads for every round/reduction round
}

out[0]=sdata[0];


}

float array_sum(const std::vector<float>& x) {
int n=x.size();
float * d_x;
float * d_out;
cudaMalloc(&d_x, n*sizeof(float));
cudaMemcpy(d_x,x.data(),n*sizeof(float),cudaMemcpyHostToDevice);//cpu to gpu
cudaMalloc(&d_out, 1*sizeof(float));
sum_kernel<<<1,256>>>(d_x,d_out,n);
cudaDeviceSynchronize();
//we need to copy result back to the cpu variable , in order to return 
float result;
cudaMemcpy(&result,d_out,1*sizeof(float),cudaMemcpyDeviceToHost);
return result;

    return 0.0f;
}
