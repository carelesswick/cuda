#include <stdio.h>

__global__ void hello_from_gpu(){
    const int bid = blockIdx.x; //获取块索引
    const int tid = threadIdx.x; //获取线程索引

    const int id = threadIdx.x + blockIdx.x * blockDim.x; //计算全局线程索引
    
    printf("Hello World from block %d, thread %d, global id %d\n", bid, tid, id);
}

int main(){
    printf("Hello World from CPU\n");
    hello_from_gpu<<<10, 5>>>(); //启动10个块，每个块5个线程
    cudaDeviceSynchronize(); //等待所有线程完成
    return 0;
}