#include <stdio.h>

//核函数
__global__ void hello_from_gpu(){
    printf("Hello World from GPU!\n");
}

int main(){
    hello_from_gpu<<<4,4>>>(); //配置线程 4*4 一共16个线程
    cudaDeviceSynchronize(); //等待GPU执行完毕
    return 0;
}