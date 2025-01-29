#include <stdlib.h>
#include <cuda.h>
#define N 1024 * 1024
#define THREADS_PER_BLOCK 512
__global__ void add(float* a, float* b, float* c,int size){
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                __shared__ float cache[THREADS_PER_BLOCK];
                int cacheIndex = threadIdx.x;

                float temp = 0.0f;
                while(tid < size){
                        temp = a[tid] * b[tid];
                        tid += blockDim.x * gridDim.x;
                }
                cache[cacheIndex] = temp;
                __syncthreads();


                for(int stride = blockDim.x / 2; stride > 0; stride = stride / 2){
                        if(cacheIndex < stride){
                                cache[cacheIndex] += cache[cacheIndex + stride];
                        }
                        __syncthreads();
                }

                if(cacheIndex == 0){
                        atomicAdd(c, cache[0]);
                }
}
void random_ints(float* x, int size){
       for(int i = 0; i < N; i++){
        x[i] = (float)rand() / (float)rand();
       }
}

int main(){
        float *a,*b,*c;
        float *d_a,*d_b, *d_c;
        int size = N * sizeof(float);

        cudaMalloc((void**) &d_a, size);
        cudaMalloc((void**) &d_b, size);
        cudaMalloc((void**) &d_c, sizeof(float));
        *c = 0.0f;
        a = (float*)malloc(size);
        random_ints(a, size);
        b = (float*)malloc(size);
        random_ints(b, size);
        c = (float*)malloc(sizeof(float));
             
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        add<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c,N);
         
        cudaMemcpy(c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
        printf("%.9f", *c);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return 0;
}
