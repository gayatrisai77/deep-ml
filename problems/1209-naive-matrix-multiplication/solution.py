#include <cuda_runtime.h>
#include <vector>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    //Mxk mul with kxN =MxN
    // thread (row, col): if row<M && col<N, C[row*N+col] = sum_k A[row*K+k]*B[k*N+col]
    int thread_id=blockDim.x*blockIdx.x+threadIdx.x;
    int total_threads=gridDim.x*blockDim.x;
    for(int i=thread_id;i<M*N;i+=total_threads){

int row=i/N;
int col=i%N;
float sum=0.0f;
for(int kdash=0;kdash<K;kdash++){
sum+=A[row*K+kdash]*B[kdash*N+col];
}
C[row*N+col]=sum;
    }




}

std::vector<float> matmul(const std::vector<std::vector<float>>& A,
                          const std::vector<std::vector<float>>& B) {

                            
                            int M=A.size();
                            int K=A[0].size();
                            int K2=B.size();
                            int N=B[0].size();
                        std::   vector<float> h_a(M*K);
                      std::  vector<float>h_b(K2*N);
                           std::  vector<float>h_c(M*N);
                            //flatten the matrices in the cpu first
for(int i=0;i<M;i++){
    for(int j=0;j<K;j++){
h_a[i*K+j]=A[i][j];

    }
}
for(int i=0;i<K2;i++){
    for(int j=0;j<N;j++){
        h_b[i*N+j]=B[i][j];
    }
}

float* d_a;
float* d_b;
float* d_c;
cudaMalloc(&d_a,M*K*sizeof(float));
cudaMalloc(&d_b,K*N*sizeof(float));
cudaMalloc(&d_c,M*N*sizeof(float));
cudaMemcpy(d_a, h_a.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);

matmul_kernel<<<1,16>>>(d_a,d_b,d_c,M,K,N);
cudaDeviceSynchronize();
cudaMemcpy(h_c.data(),d_c,M*N*sizeof(float),cudaMemcpyDeviceToHost);
return h_c;



    return {};
}
