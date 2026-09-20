#include <cuda_runtime.h>
#include <vector>

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    // load a[tid]*b[tid] (or 0) into shared memory, then tree-reduce and write out[0]
   
        __shared__ float sdata[256];
    int local_tid=threadIdx.x;//for sdata
int global_tid=blockDim.x*blockIdx.x+threadIdx.x;
if(global_tid<n){
    sdata[local_tid]=a[global_tid]*b[global_tid];
    }
    else{
        sdata[local_tid]=0.0f;
    }
    __syncthreads();
    //now we got the dot producted array, like previous question we need to sum the elemnts of the array
    //sum of elements of the array
   // int tid=blockIdx.x*blockDim.x+threadIdx.x; do we need to generate/call those threads again?
    for(int s=blockDim.x/2;s>0;s>>=1){
        //s/2 until it turns to 0 
        //basically the number of reduction rounds;
        if(local_tid<s){
sdata[local_tid]=sdata[local_tid]+sdata[local_tid+s];
        }
        __syncthreads();
    }

//as we are assuming only one block
if (local_tid == 0) {
out[0]=sdata[0];
}//since only tid =0 need to update

}

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
float* d_a;
int n=a.size();
cudaMalloc(&d_a, n*sizeof(float));
float* d_b;
cudaMalloc(&d_b, n*sizeof(float));
cudaMemcpy(d_a,a.data(),n*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(d_b,b.data(),n*sizeof(float),cudaMemcpyHostToDevice);
float * out;
cudaMalloc(&out,1*sizeof(float));
dot_kernel<<<1,256>>>(d_a,d_b,out,n);

cudaDeviceSynchronize();

float result;
cudaMemcpy(&result,out, 1*sizeof(float), cudaMemcpyDeviceToHost);//gpu to cpu
return result;

    return 0.0f;
}
