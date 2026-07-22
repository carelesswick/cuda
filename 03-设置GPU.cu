#include <stdio.h>

int main(){

    //检查GPU数量
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found or error occurred: %s\n", cudaGetErrorString(error));
        return -1;
    }
    else{
        printf("Found %d CUDA devices\n", deviceCount);
    }


    //获取设备属性
    int Dev = 0;
    error = cudaSetDevice(Dev);
    if (error != cudaSuccess) {
        printf("Failed to set device %d: %s\n", Dev, cudaGetErrorString(error));
        return -1;
    }
    else{
        printf("Set device %d\n", Dev);
    }


    return 0;
}