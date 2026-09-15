# Author: even
"""cuFFT reference programs with literal plans consumed by the restricted adapter."""

from .config import workspace, remote_root, npu_environment, connect_ssh
import math
from .fft_ir import import_cuda


def render(shape, batch):
    rank = len(shape)
    if rank > 1 and batch != 1:
        raise ValueError("this source subset uses batch=1 for ND")
    plan = "cufftPlan%dd(&plan, %s, CUFFT_C2C%s)" % (
        rank,
        ", ".join(map(str, shape)),
        ", %d" % batch if rank == 1 else "",
    )
    src = (
        CUDA.replace("@PLAN@", plan)
        .replace("@TOTAL@", str(math.prod(shape) * batch))
        .replace("@NORM@", str(math.prod(shape)))
    )
    return src


def source(shape, batch):
    src = render(shape, batch)
    c = import_cuda(src)
    assert c.shape == tuple(shape) and c.batch == batch
    return src, c


CUDA = r"""
// SAGE_CONTRACT {"pipeline":"fft_complexmul_ifft","normalization":"1/N","layout":"interleaved_row_major"}
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}
void cf(cufftResult e){if(e!=CUFFT_SUCCESS){fprintf(stderr,"CUFFT %d\n",e);exit(3);}}
__global__ void multiply(cufftComplex*z,const cufftComplex*h){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<@TOTAL@){auto a=z[k],b=h[k];z[k]={a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x};}}
__global__ void normalize(cufftComplex*y){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<@TOTAL@){y[k].x/=@NORM@;y[k].y/=@NORM@;}}
int main(int argc,char**argv){if(argc<3)return 1;ck(cudaSetDevice(0));
size_t bytes=@TOTAL@*sizeof(cufftComplex);std::vector<cufftComplex> a(@TOTAL@),b(@TOTAL@);
std::ifstream fx(std::string(argv[1])+"/x.bin",std::ios::binary),fh(std::string(argv[1])+"/h.bin",std::ios::binary);if(!fx.read((char*)a.data(),bytes)||!fh.read((char*)b.data(),bytes))return 4;
cufftComplex *x,*h,*z,*y;ck(cudaMalloc(&x,bytes));ck(cudaMalloc(&h,bytes));ck(cudaMalloc(&z,bytes));ck(cudaMalloc(&y,bytes));ck(cudaMemcpy(x,a.data(),bytes,cudaMemcpyHostToDevice));ck(cudaMemcpy(h,b.data(),bytes,cudaMemcpyHostToDevice));
cufftHandle plan;cf(@PLAN@);
auto run=[&](){cf(cufftExecC2C(plan, x, z, CUFFT_FORWARD));multiply<<<(@TOTAL@+255)/256,256>>>(z,h);ck(cudaGetLastError());cf(cufftExecC2C(plan, z, y, CUFFT_INVERSE));normalize<<<(@TOTAL@+255)/256,256>>>(y);ck(cudaGetLastError());};
run();ck(cudaDeviceSynchronize());ck(cudaMemcpy(a.data(),y,bytes,cudaMemcpyDeviceToHost));std::ofstream f(argv[2],std::ios::binary);f.write((char*)a.data(),bytes);f.close();
for(int i=0;i<3;i++)run();ck(cudaDeviceSynchronize());cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));printf("{\"samples_us\":[");
for(int g=0;g<10;g++){ck(cudaEventRecord(start));for(int i=0;i<3;i++)run();ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));printf("%s%.9f",g?",":"",ms*1000/3);}printf("]}\n");
cudaEventDestroy(start);cudaEventDestroy(end);cufftDestroy(plan);cudaFree(x);cudaFree(h);cudaFree(z);cudaFree(y);return 0;}
"""
