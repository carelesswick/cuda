#include <stdlib.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREAD_PER_BLOCK 256                    //每一个block的线程数（一般都是设置成256）

__global__ void reduce(float* d_input,float* d_output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;//全局索引
    int tid = threadIdx.x; //块内线程索引

    // float * res = nullptr; 不能这么写树状 reduce 的临时数组，必须是__shared__共享内存，不能随便定义普通指针。
    // 还有一些CPU端的函数 一般这些封装的函数 都是给CPU用的，cuda内部不可使用

    //每个块是并行的 因此 需要对每个块内及加规则 因为每个块都是一样的逻辑 想通了这点 就知道是对blockDim.x进行循环
    for(int i = 1; i < blockDim.x; i*=2){
        if(threadIdx.x % (2*i) == 0){
            d_input[index] += d_input[index + i];
        }
        __syncthreads();
    }
    if(tid == 0){
        d_output[blockIdx.x] = d_input[index];

    }
}

/////////////////////////////////////////////////////////////
//更容易理解的方法
/////////////////////////////////////////////////////////////
__global__ void reduce(float* d_input,float* d_output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;//全局索引
    int tid = threadIdx.x; //块内线程索引
    //指针 + 整数偏移（确保在每一个块开始的时候都是在 threadIDx.x = 0的起始位置）
    float *input_begin = blockIdx.x * blockDim.x + d_input;
    for(int i = 1; i < blockDim.x; i*=2){
        if(threadIdx.x % (2*i) == 0){
            input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        d_output[blockIdx.x] = input_begin[0];
    }
}


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
