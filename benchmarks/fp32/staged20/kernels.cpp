#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void fft_step_0(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,64);pipe.InitBuffer(rb,64);pipe.InitBuffer(hb,64);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*1;line<64;line+=GetBlockNum()*1){int base=(line/1)*8+line%1;
DataCopy(raw,x[2*base],16);PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;int rev=0;for(int b=0;b<3;b++){rev=(rev<<1)|(p&1);p>>=1;}p=rev;for(int l=0;l<1;l++){v.SetValue(2*(k*1+l),raw.GetValue(2*(p*1+l)));v.SetValue(2*(k*1+l)+1,raw.GetValue(2*(p*1+l)+1));}}
for(int b=0;b<8;b+=2)for(int l=0;l<1;l++){int u=2*((b+0)*1+l),q=u+2;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,16);PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_1(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,64);pipe.InitBuffer(rb,64);pipe.InitBuffer(hb,64);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*1;line<64;line+=GetBlockNum()*1){int base=(line/1)*8+line%1;
DataCopy(raw,x[2*base],16);PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<1;l++){v.SetValue(2*(k*1+l),raw.GetValue(2*(p*1+l)));v.SetValue(2*(k*1+l)+1,raw.GetValue(2*(p*1+l)+1));}}
for(int b=0;b<8;b+=4)for(int l=0;l<1;l++){int u=2*((b+0)*1+l),q=u+4;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=4)for(int l=0;l<1;l++){int u=2*((b+1)*1+l),q=u+4;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(-1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(-1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,16);PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_2(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,64);pipe.InitBuffer(rb,64);pipe.InitBuffer(hb,64);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*1;line<64;line+=GetBlockNum()*1){int base=(line/1)*8+line%1;
DataCopy(raw,x[2*base],16);PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<1;l++){v.SetValue(2*(k*1+l),raw.GetValue(2*(p*1+l)));v.SetValue(2*(k*1+l)+1,raw.GetValue(2*(p*1+l)+1));}}
for(int b=0;b<8;b+=8)for(int l=0;l<1;l++){int u=2*((b+0)*1+l),q=u+8;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<1;l++){int u=2*((b+1)*1+l),q=u+8;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(7.071067811865e-01f)*br-(-7.071067811865e-01f)*bi,ti=(7.071067811865e-01f)*bi+(-7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<1;l++){int u=2*((b+2)*1+l),q=u+8;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(-1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(-1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<1;l++){int u=2*((b+3)*1+l),q=u+8;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(-7.071067811865e-01f)*br-(-7.071067811865e-01f)*bi,ti=(-7.071067811865e-01f)*bi+(-7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,16);PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_3(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/8)*64+line%8;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*8)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;int rev=0;for(int b=0;b<3;b++){rev=(rev<<1)|(p&1);p>>=1;}p=rev;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=2)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+16;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*8)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_4(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/8)*64+line%8;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*8)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=4)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+32;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=4)for(int l=0;l<8;l++){int u=2*((b+1)*8+l),q=u+32;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(-1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(-1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*8)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_5(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/8)*64+line%8;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*8)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+1)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(7.071067811865e-01f)*br-(-7.071067811865e-01f)*bi,ti=(7.071067811865e-01f)*bi+(-7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+2)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(-1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(-1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+3)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(-7.071067811865e-01f)*br-(-7.071067811865e-01f)*bi,ti=(-7.071067811865e-01f)*bi+(-7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*8)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_6(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/64)*512+line%64;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*64)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;int rev=0;for(int b=0;b<3;b++){rev=(rev<<1)|(p&1);p>>=1;}p=rev;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=2)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+16;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*64)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_7(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/64)*512+line%64;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*64)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=4)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+32;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=4)for(int l=0;l<8;l++){int u=2*((b+1)*8+l),q=u+32;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(-1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(-1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*64)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_8(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/64)*512+line%64;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*64)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(-0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(-0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+1)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(7.071067811865e-01f)*br-(-7.071067811865e-01f)*bi,ti=(7.071067811865e-01f)*bi+(-7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+2)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(-1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(-1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+3)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(-7.071067811865e-01f)*br-(-7.071067811865e-01f)*bi,ti=(-7.071067811865e-01f)*bi+(-7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*64)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_9(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,hb; pipe.InitBuffer(vb,256);pipe.InitBuffer(hb,256); auto v=vb.Get<float>();auto w=hb.Get<float>();
for(int base=GetBlockIdx()*32;base<512;base+=GetBlockNum()*32){DataCopy(v,x[2*base],64);DataCopy(w,h[2*base],64);PipeBarrier<PIPE_ALL>();for(int k=0;k<32;k++){float r=v.GetValue(2*k),i=v.GetValue(2*k+1);float hr=w.GetValue(2*k),hi=w.GetValue(2*k+1);float rr=r*hr-i*hi;i=r*hi+i*hr;r=rr;v.SetValue(2*k,r);v.SetValue(2*k+1,i);}PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,64);PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_10(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,64);pipe.InitBuffer(rb,64);pipe.InitBuffer(hb,64);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*1;line<64;line+=GetBlockNum()*1){int base=(line/1)*8+line%1;
DataCopy(raw,x[2*base],16);PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;int rev=0;for(int b=0;b<3;b++){rev=(rev<<1)|(p&1);p>>=1;}p=rev;for(int l=0;l<1;l++){v.SetValue(2*(k*1+l),raw.GetValue(2*(p*1+l)));v.SetValue(2*(k*1+l)+1,raw.GetValue(2*(p*1+l)+1));}}
for(int b=0;b<8;b+=2)for(int l=0;l<1;l++){int u=2*((b+0)*1+l),q=u+2;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,16);PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_11(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,64);pipe.InitBuffer(rb,64);pipe.InitBuffer(hb,64);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*1;line<64;line+=GetBlockNum()*1){int base=(line/1)*8+line%1;
DataCopy(raw,x[2*base],16);PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<1;l++){v.SetValue(2*(k*1+l),raw.GetValue(2*(p*1+l)));v.SetValue(2*(k*1+l)+1,raw.GetValue(2*(p*1+l)+1));}}
for(int b=0;b<8;b+=4)for(int l=0;l<1;l++){int u=2*((b+0)*1+l),q=u+4;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=4)for(int l=0;l<1;l++){int u=2*((b+1)*1+l),q=u+4;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,16);PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_12(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,64);pipe.InitBuffer(rb,64);pipe.InitBuffer(hb,64);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*1;line<64;line+=GetBlockNum()*1){int base=(line/1)*8+line%1;
DataCopy(raw,x[2*base],16);PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<1;l++){v.SetValue(2*(k*1+l),raw.GetValue(2*(p*1+l)));v.SetValue(2*(k*1+l)+1,raw.GetValue(2*(p*1+l)+1));}}
for(int b=0;b<8;b+=8)for(int l=0;l<1;l++){int u=2*((b+0)*1+l),q=u+8;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<1;l++){int u=2*((b+1)*1+l),q=u+8;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(7.071067811865e-01f)*br-(7.071067811865e-01f)*bi,ti=(7.071067811865e-01f)*bi+(7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<1;l++){int u=2*((b+2)*1+l),q=u+8;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<1;l++){int u=2*((b+3)*1+l),q=u+8;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(-7.071067811865e-01f)*br-(7.071067811865e-01f)*bi,ti=(-7.071067811865e-01f)*bi+(7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,16);PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_13(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/8)*64+line%8;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*8)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;int rev=0;for(int b=0;b<3;b++){rev=(rev<<1)|(p&1);p>>=1;}p=rev;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=2)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+16;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*8)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_14(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/8)*64+line%8;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*8)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=4)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+32;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=4)for(int l=0;l<8;l++){int u=2*((b+1)*8+l),q=u+32;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*8)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_15(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/8)*64+line%8;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*8)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+1)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(7.071067811865e-01f)*br-(7.071067811865e-01f)*bi,ti=(7.071067811865e-01f)*bi+(7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+2)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+3)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(-7.071067811865e-01f)*br-(7.071067811865e-01f)*bi,ti=(-7.071067811865e-01f)*bi+(7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*8)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_16(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/64)*512+line%64;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*64)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;int rev=0;for(int b=0;b<3;b++){rev=(rev<<1)|(p&1);p>>=1;}p=rev;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=2)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+16;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*64)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_17(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/64)*512+line%64;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*64)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=4)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+32;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=4)for(int l=0;l<8;l++){int u=2*((b+1)*8+l),q=u+32;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*64)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_18(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,512);pipe.InitBuffer(rb,512);pipe.InitBuffer(hb,512);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();
for(int line=GetBlockIdx()*8;line<64;line+=GetBlockNum()*8){int base=(line/64)*512+line%64;
for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(base+k*64)],16);}PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){int p=k;for(int l=0;l<8;l++){v.SetValue(2*(k*8+l),raw.GetValue(2*(p*8+l)));v.SetValue(2*(k*8+l)+1,raw.GetValue(2*(p*8+l)+1));}}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+0)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(1.000000000000e+00f)*br-(0.000000000000e+00f)*bi,ti=(1.000000000000e+00f)*bi+(0.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+1)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(7.071067811865e-01f)*br-(7.071067811865e-01f)*bi,ti=(7.071067811865e-01f)*bi+(7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+2)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(6.123233995737e-17f)*br-(1.000000000000e+00f)*bi,ti=(6.123233995737e-17f)*bi+(1.000000000000e+00f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
for(int b=0;b<8;b+=8)for(int l=0;l<8;l++){int u=2*((b+3)*8+l),q=u+64;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=(-7.071067811865e-01f)*br-(7.071067811865e-01f)*bi,ti=(-7.071067811865e-01f)*bi+(7.071067811865e-01f)*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}
PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++){DataCopy(y[2*(base+k*64)],v[k*16],16);}PipeBarrier<PIPE_ALL>();}}

extern "C" __global__ __aicore__ void fft_step_19(GM_ADDR input,GM_ADDR weight,GM_ADDR output){
GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);
TPipe pipe; TBuf<TPosition::VECCALC> vb,hb; pipe.InitBuffer(vb,256);pipe.InitBuffer(hb,256); auto v=vb.Get<float>();auto w=hb.Get<float>();
for(int base=GetBlockIdx()*32;base<512;base+=GetBlockNum()*32){DataCopy(v,x[2*base],64);PipeBarrier<PIPE_ALL>();for(int k=0;k<32;k++){float r=v.GetValue(2*k),i=v.GetValue(2*k+1);r*=1.953125000000e-03f;i*=1.953125000000e-03f;v.SetValue(2*k,r);v.SetValue(2*k+1,i);}PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,64);PipeBarrier<PIPE_ALL>();}}
