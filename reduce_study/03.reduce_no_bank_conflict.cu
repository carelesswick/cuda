#include <stdlib.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREAD_PER_BLOCK 256                    //每一个block的线程数（一般都是设置成256）

__global__ void reduce(float* d_input,float* d_output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;//全局索引
    int tid = threadIdx.x; 
    __shared__ float sdata[THREAD_PER_BLOCK];
    sdata[threadIdx.x] = d_input[index];
    __syncthreads();//等待搬运完数据

    
    for(int i = blockDim.x / 2; i > 0; i /= 2){
        if(threadIdx.x < i){

            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(tid == 0){
        d_output[blockIdx.x] = sdata[0];

    }
}

/////////////////////////////////////////////////////////////
//更容易理解的方法
/////////////////////////////////////////////////////////////
// __global__ void reduce(float* d_input,float* d_output)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int tid = threadIdx.x; 
    
//     //共享内存
//      __shared__ float sdata[THREAD_PER_BLOCK]; 

//     //指针 + 整数偏移（确保在每一个块开始的时候都是在 threadIDx.x = 0的起始位置）
//     float *input_begin = blockIdx.x * blockDim.x + d_input;
    
//     //记住并行的思想 搬运数据的时候 block之间是并行搬运的
//     sdata[threadIdx.x] = input_begin[threadIdx.x];
//     __syncthreads();//等待搬运完数据

//     for(int i = 1; i < blockDim.x; i*=2){
//         if(threadIdx.x < (blockDim.x / (2*i))){
//             //因为线程要挨在一起 所以呢索引会改变
//             int mid_index =threadIdx.x * 2 * i;
//             sdata[mid_index] += sdata[mid_index + i];
//         }
//         // if(threadIdx.x % (2*i) == 0){
//         //     sdata[threadIdx.x] += sdata[threadIdx.x + i];
//         // }
//         __syncthreads();
//     }
//     if (threadIdx.x == 0)
//     {
//         d_output[blockIdx.x] = sdata[threadIdx.x];
//     }
// }


bool check(float *out,float *res,int n){
    for(int i = 0 ;i < n;++i){
        if(fabs(out[i] - res[i]) > 0.0005){
            return false;
        }

    }
    return true;
}


int main(){


    const int N = 32*1024*1024;
    size_t size = N*sizeof(float);
    float *h_input = (float*)malloc(size);
    float *d_input;

    cudaMalloc((void **)&d_input,size);

    int block_num = N / THREAD_PER_BLOCK;       //根据运算数量得到block的数量
    float *h_out = (float*)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
    float *d_out;
    cudaMalloc((void**)&d_out,(N / THREAD_PER_BLOCK) * sizeof(float));
    float *h_res = (float*)malloc((N / THREAD_PER_BLOCK) * sizeof(float));

    //cpu端随机生成h_input数组
    for(int i = 0; i < N; ++i){
        h_input[i] = 2.0*(float)drand48() - 1.0;//drand48()属于stdlib，返回[0.0,1.0)中的非负双精度浮点值
    }

    //cpu计算golden参考h_res
    for(int i = 0; i < block_num; ++i){
        float cur = 0;
        for(int j = 0; j < THREAD_PER_BLOCK;++j){
            cur+=h_input[i * THREAD_PER_BLOCK + j];
        }
        h_res[i] = cur;
    }

    cudaMemcpy(d_input,h_input,size,cudaMemcpyHostToDevice);

    dim3 GRid (block_num,1);
    dim3 Block (THREAD_PER_BLOCK,1);

    reduce<<<GRid,Block>>>(d_input,d_out);

    //将GPU的输出拷回到CPU
    cudaMemcpy(h_out,d_out,block_num * sizeof(float),cudaMemcpyDeviceToHost);

    if (check(h_out,h_res,block_num)){
        printf("the ans is right! \n");
    }
    else{
        printf("thr answer is wrong! \n");
        for(int i = 0;i < block_num; ++i){
            printf("%lf",h_out[i]);
        }
        printf("\n");
    }


    cudaFree(d_out);
    cudaFree(d_input);
    free(h_res);
    free(h_out);
    free(h_input);


    return 0;

}
