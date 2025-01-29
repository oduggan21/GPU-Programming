#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#define N 1048576
#define THREADS_PER_BLOCK 512

__global__ void add(float* a, float* b, double* c,int size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double cache[THREADS_PER_BLOCK];
    int cacheIndex = threadIdx.x;
    double temp = 0.0;
    while(tid < size){
        temp += (double)a[tid] * (double)b[tid];
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
        c[blockIdx.x] = cache[0];
    }
}

__global__ void addBlocks(double* c, int numBlocks){
    __shared__ double cache2[THREADS_PER_BLOCK];
    int tid2 = threadIdx.x;
    double temp2 = 0.0;
    for(int i = tid2; i < numBlocks; i += blockDim.x){
        temp2 += c[i];
    }
    cache2[tid2] = temp2;
    __syncthreads();
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid2 < stride){
            cache2[tid2] += cache2[tid2 + stride];
        }
        __syncthreads();
    }
    if(tid2 == 0){
        c[0] = cache2[0];
    }
}

long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

long long stop_timer(long long start_time,const char *name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    printf("%s: %.5f sec\n", name, ((float)(end_time - start_time)) / (1000 * 1000));
    return end_time - start_time;
}

void random_ints(float* x, int size){
    for(int i = 0; i < N; i++){
        x[i] = (float)rand() / (float)rand();
    }
}

double CPU_big_dot(float* A, float* B, int n){
    double c = 0.0;
    for(int i =0; i < n; i++){
        c += (double)A[i] * (double)B[i];
    }
    return c;
}

double GPU_big_dot(float* A, float* B, int n, long long *alloc_transfer_time, long long *kernel_time, long long *transfer_back_time){
    double result;
    float *d_a,*d_b;
    double* d_c;
    int size  = N * sizeof(float);
    int size_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    long long start, stop;

    start = start_timer();
    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, (size_blocks * sizeof(double)));
    cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);
    stop = stop_timer(start, "Alloc and Host to Device");
    *alloc_transfer_time = stop;

    start = start_timer();
    add<<<size_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    addBlocks<<<1, THREADS_PER_BLOCK>>>(d_c, size_blocks);
    cudaDeviceSynchronize();
    stop = stop_timer(start, "Kernel Execution");
    *kernel_time = stop;

    start = start_timer();
    cudaMemcpy(&result, d_c, sizeof(double), cudaMemcpyDeviceToHost);
    stop = stop_timer(start, "Device to Host");
    *transfer_back_time = stop;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return result;
}

int main(){
    float *a,*b;
    int size = N * sizeof(float);
    int N_size = N*1;
    a = (float*)malloc(size);
    random_ints(a, size);
    b = (float*)malloc(size);
    random_ints(b, size);
    const char* gpuFunction = "GPU_FUNCTION";
    const char* cpuFunction = "CPU_FUNCTION";
    long long alloc_transfer_time, kernel_time, transfer_back_time;
    long long start_time = start_timer();
    double c_gpu = GPU_big_dot(a, b, N_size, &alloc_transfer_time, &kernel_time, &transfer_back_time);
    stop_timer(start_time, gpuFunction);    
    start_time = start_timer();
    double c_cpu = CPU_big_dot(a, b, N_size);
    long long cpu_time = stop_timer(start_time, cpuFunction);
    double diff = fabs(c_gpu - c_cpu);
    printf("Performance\n");
    printf("CPU Time: %.5f sec\n", ((float)cpu_time) / 1000000);
    printf("GPU memory allocation and data transfer: %.5f sec\n", ((float)alloc_transfer_time) / 1000000);
    printf("GPU kernel execution time : %.5f sec\n", ((float)kernel_time) / 1000000);
    printf("GPU data transfer back to CPU: %.5f sec\n", ((float)transfer_back_time) / 1000000);
    double Tgpu = ((float)(alloc_transfer_time + kernel_time + transfer_back_time)) / 1000000;
    double speedup = ((float)cpu_time) / Tgpu / 1000000;
    printf("Total GPU time: %.5f sec\n", Tgpu);
    printf("Speedup: %.2f\n", speedup);
    double speedup2 = (double)cpu_time / (double)kernel_time;
    printf("Speedup only on kernel time: %.2f\n", speedup2);
    free(a);
    free(b);
    return 0;
}
