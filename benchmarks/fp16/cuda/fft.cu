#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <functional>
#include <cstdio>
#include <cstdlib>
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}
void cf(cufftResult e){if(e!=CUFFT_SUCCESS){fprintf(stderr,"CUFFT %d\n",e);exit(3);}}
__global__ void multiply(__half2*z,const __half2*h){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<512){auto a=z[k],b=h[k];z[k]=__halves2half2(__hsub(__hmul(a.x,b.x),__hmul(a.y,b.y)),__hadd(__hmul(a.x,b.y),__hmul(a.y,b.x)));}}
__global__ void normalize(__half2*y){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<512)y[k]=__hmul2(y[k],__float2half2_rn(1.0f/512));}
template<int K> __global__ void empty_kernel(__half2*x,__half2*h,__half2*y){}
int main(int argc,char**argv){
 if(argc<5||argc>6){fprintf(stderr,"usage: fft INPUT_DIR OUTPUT ITERATIONS MODE [DEVICE=0]\n");return 1;}int reps=atoi(argv[3]);std::string mode=argv[4];bool empty=mode.find("empty")!=std::string::npos;
 int graphSize=mode.find("graph100")!=std::string::npos?100:(mode.find("graph1")!=std::string::npos?1:0);
 if(reps<1||reps>100000||(graphSize&&reps%graphSize))return 4;
 if(mode!="direct"&&mode!="graph1"&&mode!="graph100"&&mode!="empty")return 1;
 int device=argc>5?atoi(argv[5]):0;ck(cudaSetDevice(device));cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 size_t bytes=512*sizeof(__half2);std::vector<__half2> a(512),b(512);
 std::ifstream fx(std::string(argv[1])+"/x.bin",std::ios::binary),fh(std::string(argv[1])+"/h.bin",std::ios::binary);
 if(!fx.read((char*)a.data(),bytes)||!fh.read((char*)b.data(),bytes)||fx.peek()!=EOF||fh.peek()!=EOF)return 5;
 __half2 *x,*h,*z,*y;ck(cudaMalloc(&x,bytes));ck(cudaMalloc(&h,bytes));ck(cudaMalloc(&z,bytes));ck(cudaMalloc(&y,bytes));
 ck(cudaMemcpy(x,a.data(),bytes,cudaMemcpyHostToDevice));ck(cudaMemcpy(h,b.data(),bytes,cudaMemcpyHostToDevice));ck(cudaMemcpy(y,a.data(),bytes,cudaMemcpyHostToDevice));
 cufftHandle plan;cf(cufftCreate(&plan));long long dims[3]={8,8,8};size_t worksize=0;cf(cufftXtMakePlanMany(plan,3,dims,nullptr,1,512,CUDA_C_16F,nullptr,1,512,CUDA_C_16F,1,&worksize,CUDA_C_16F));cf(cufftSetStream(plan,stream));
 auto pipeline=[&](){
  if(empty){
   empty_kernel<0><<<1,128,4096,stream>>>(x,h,y);empty_kernel<1><<<1,128,4096,stream>>>(x,h,y);empty_kernel<2><<<2,dim3(2,32),2048,stream>>>(x,h,y);
   empty_kernel<3><<<2,256,0,stream>>>(x,h,y);
   empty_kernel<4><<<1,128,4096,stream>>>(x,h,y);empty_kernel<5><<<1,128,4096,stream>>>(x,h,y);empty_kernel<6><<<2,dim3(2,32),2048,stream>>>(x,h,y);
   empty_kernel<7><<<2,256,0,stream>>>(x,h,y);ck(cudaGetLastError());
  }else{cf(cufftXtExec(plan,x,z,CUFFT_FORWARD));multiply<<<2,256,0,stream>>>(z,h);ck(cudaGetLastError());cf(cufftXtExec(plan,z,y,CUFFT_INVERSE));normalize<<<2,256,0,stream>>>(y);ck(cudaGetLastError());}
 };
 auto save=[&](std::string name){ck(cudaMemcpy(a.data(),y,bytes,cudaMemcpyDeviceToHost));std::ofstream f(name,std::ios::binary);f.write((char*)a.data(),bytes);f.close();if(!f){fprintf(stderr,"output write failed\n");exit(5);}};
 pipeline();ck(cudaStreamSynchronize(stream));save(argv[2]);for(int w=0;w<1000;w++)pipeline();ck(cudaStreamSynchronize(stream));
 cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
 if(graphSize){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int q=0;q<graphSize;q++)pipeline();ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));}
 std::vector<double> total,submit,drain;
 for(int g=0;g<11;g++){
  auto t0=std::chrono::steady_clock::now();
  if(graphSize){for(int j=0;j<reps/graphSize;j++)ck(cudaGraphLaunch(exec,stream));}else for(int j=0;j<reps;j++)pipeline();
  ck(cudaStreamSynchronize(stream));auto t2=std::chrono::steady_clock::now();
  total.push_back(std::chrono::duration<double,std::micro>(t2-t0).count()/reps);
 }
 save(std::string(argv[2])+".after");
 auto emit=[&](const char*k,const std::vector<double>&v){printf("\"%s\":[",k);for(size_t j=0;j<v.size();j++)printf("%s%.9f",j?",":"",v[j]);printf("]");};
 printf("{\"timing_method\":\"host_batch_submit_sync\",\"unit\":\"us_per_pipeline\",\"precision\":\"fp16\",\"warmup\":1000,\"groups\":11,\"iterations\":%d,\"mode\":\"%s\",\"graph_pipelines_per_launch\":%d,\"graph_warmup_pipelines\":%d,\"device_index\":%d,",reps,mode.c_str(),graphSize,graphSize,device);emit("samples_us",total);printf(",");emit("submit_us",submit);printf(",");emit("drain_us",drain);printf("}\n");
 if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);cufftDestroy(plan);cudaFree(x);cudaFree(h);cudaFree(z);cudaFree(y);cudaStreamDestroy(stream);
}
