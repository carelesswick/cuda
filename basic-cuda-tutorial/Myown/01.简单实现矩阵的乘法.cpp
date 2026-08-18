#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
//矩阵乘法的关键就是要知道 矩阵是在内存中 将每一行 按末尾顺序排在一起进行存储的。这个就是索引的关键
void matrixMulCPU(float *C, const float *A, const float *B,
                    unsigned int wA,unsigned int wC, unsigned int hC)
{
    unsigned int hA = hC;
    unsigned int hB = wA;
    unsigned int wB = wC;

    
    for(int i = 0; i < hA ; ++i){
        for(int j = 0 ; j < wB ; ++j){
            float sum = 0;
            for(int k = 0; k < hB ; ++k){
                sum += A[i*wA+k]*B[k*wB+j];
            }
            C[i*wB+j] = sum;
        }
        
    }


}


//之前上一个对于B矩阵缓存不友好 进行改进
void matrixMulCPU_imp(float *C, const float *A, const float *B,
                    unsigned int wA,unsigned int wC, unsigned int hC)
{
    unsigned int hA = hC;
    unsigned int hB = wA;
    unsigned int wB = wC;

    
    for(int i = 0; i < hA ; ++i){
        for(int j = 0 ; j < hB ; ++j){
            float sum = 0;
            for(int k = 0; k < wB ; ++k){
                sum += A[i*wA+j]*B[j*wB+k];
            }
            C[i*wB+k] = sum;
        }
        
    }


}

int main()
{
    int hA = 10, wA = 10, hB = 10, wB = 10, hC = 10, wC = 10; 
    std::vector<float> matA(hA * wA, 2.1f);
    std::vector<float> matB(hB * wB, 3.2f);
    std::vector<float> matC(hC * wC, 0.0f);

    matrixMulCPU(matC.data(),matA.data(),matB.data(),wA,wC,hC);
    for (float v : matC)
    {
        std::cout << v << " ";
    }
    std::cout << "\n";



    return 0;
}