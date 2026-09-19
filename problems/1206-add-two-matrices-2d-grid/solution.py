#include <cuda_runtime.h>
#include <vector>

__global__ void matadd_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    // Compute (row, col) from the 2D thread index; guard row<rows && col<cols;
    // C[row*cols + col] = A[row*cols + col] + B[row*cols + col];
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows&&col<cols){
        int index=row*cols+col;
        C[index]=A[index]+B[index];
    }
}

std::vector<float> matrix_add(const std::vector<std::vector<float>>& A,
                              const std::vector<std::vector<float>>& B) {
    // flatten A and B to row-major 1D, run the kernel, return C flattened
 float*d_a;
float*d_b;
float*d_out;

int rows=A.size();
int cols=A[0].size();
int n=rows*cols;
cudaMalloc(&d_a, n*sizeof(float));
cudaMalloc(&d_b, n*sizeof(float));
cudaMalloc(&d_out, n*sizeof(float));

std::vector<float>flat_A;
std::vector<float>flat_B;
for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
        flat_A.push_back(A[i][j]);
        flat_B.push_back(B[i][j]);
    }
}


cudaMemcpy(d_a, flat_A.data(), n*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, flat_B.data(), n*sizeof(float), cudaMemcpyHostToDevice);


   dim3 block(16, 16);

    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );


matadd_kernel<<< grid,block >>>(d_a,d_b,d_out,rows,cols);
cudaDeviceSynchronize();//gpu syn.

  std::  vector<float>result(n);
    cudaMemcpy(result.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
    return result;
    return {};
}
