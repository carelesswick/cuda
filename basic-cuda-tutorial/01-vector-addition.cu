#include <stdio.h>
#include <stdlib.h>

//错误检查的宏定义
//定义宏时使用 do-while(0) 循环包裹，可以确保宏在使用时像一个单独的语句一样行为良好，特别是在 if-else 语句中使用时，不会因为语法错误导致逻辑分支混乱。
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel function to add two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    // Get the unique thread ID, which is the index in the vector
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (i < numElements) {
        C[i] = A[i] * B[i] + C[i];
    }
}

void vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; i++) {
        C[i] = A[i] * B[i] + C[i];
    }
}

int main() {
    // Vector size and memory size
    int numElements = 50000000;
    size_t size = numElements * sizeof(float);
    
    printf("Vector addition of %d elements\n", numElements);
    
    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // 初始化数组 他的值在 0 到 1 之间
    // ran()是C标准库函数，返回一个0到RAND_MAX之间的随机整数。通过将其除以RAND_MAX，可以将其归一化为0到1之间的浮点数。
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;  // 初始化 C，因为新公式要读 C[i]
    }
    
    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_A, size));

    CUDA_CHECK(cudaMalloc((void **)&d_B, size));

    CUDA_CHECK(cudaMalloc((void **)&d_C, size));
    
    // Copy data from host to device
    //cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)

    clock_t h_d_begin = clock(); // 记录开始时间
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));  // 把 C 的初始值也传到 GPU
    clock_t h_d_end = clock(); // 记录结束时间
    double h_d_time = (double)(h_d_end - h_d_begin) / CLOCKS_PER_SEC;
    printf("Host to Device transfer time: %.3f ms\n", h_d_time * 1000.0); // 将秒转换为毫秒

    //性能测试初始化
    cudaEvent_t start, stop;//定义两个事件变量

    CUDA_CHECK(cudaEventCreate(&start));

    CUDA_CHECK(cudaEventCreate(&stop));
    //记录事件，开始计时
    CUDA_CHECK(cudaEventRecord(start));

    // Copy data from host to device

    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    

    //性能测试结束

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);


    CUDA_CHECK(cudaEventDestroy(start));

    CUDA_CHECK(cudaEventDestroy(stop));



    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // 保存原始 h_C 用于验证（因为 GPU 结果会覆盖它）
    float *h_C_orig = (float *)malloc(size);
    memcpy(h_C_orig, h_C, size);

    // Copy result back to host
    clock_t d_h_begin = clock(); // 记录开始时间
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    clock_t d_h_end = clock(); // 记录结束时间
    double d_h_time = (double)(d_h_end - d_h_begin) / CLOCKS_PER_SEC; // 将秒转换为毫秒
    printf("Device to Host transfer time: %.3f ms\n", d_h_time * 1000.0);

    // Verify the result: check that GPU's h_C == A * B + original_C
    for (int i = 0; i < numElements; ++i) {
        float expected = h_A[i] * h_B[i] + h_C_orig[i];
        if (fabs(expected - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    free(h_C_orig);
    printf("Test PASSED\n");

    clock_t start_cpu = clock(); // 记录开始时间

    vectorAddCPU(h_A, h_B, h_C, numElements);
    clock_t end_cpu = clock(); // 记录结束时间
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    
    printf("CPU execution time: %.3f ms\n", cpu_time * 1000.0); // 将秒转换为毫秒
    // Free device memory

    CUDA_CHECK(cudaFree(d_A));

    CUDA_CHECK(cudaFree(d_B));

    CUDA_CHECK(cudaFree(d_C));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done\n");
    return 0;
} 