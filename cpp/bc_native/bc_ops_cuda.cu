#include "bc_ops.h"
#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace bc_native {
__global__ void props_kernel(int64_t n,const double*s,const bool*m,const double*t,double*o,Params p){
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(blockDim.x)*gridDim.x)props_cell(i,n,s,m,t,o,p);
}
__global__ void advance_kernel(int64_t n,const double*s,const bool*m,const bool*a,const double*tr,const double*h,const double*t,double*o,double*d,Params p){
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(blockDim.x)*gridDim.x)advance_cell(i,n,s,m,a,tr,h,t,o,d,p);
}
__global__ void contacts_kernel(int64_t m,int64_t n,const int64_t*ci,const bool*an,const double*s,const double*tr,const double*v,const bool*a,double*o,Params p){
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<m;i+=int64_t(blockDim.x)*gridDim.x)contact_cell(i,m,n,ci,an,s,tr,v,a,o,p);
}
__global__ void current_kernel(int64_t n,const double*s,const double*tr,const bool*m,double*o,Params p){
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(blockDim.x)*gridDim.x)current_cell(i,n,s,tr,m,o,p);
}
__global__ void stencil_kernel(int64_t n,const double*c,const double*x,double*y,int side){
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(blockDim.x)*gridDim.x)stencil_cell(i,n,c,x,y,side);
}
static int blocks(int64_t n){return int(std::min<int64_t>((n+255)/256,65535));}
Tensor props_cuda(const Tensor&s,const Tensor&m,const Tensor&t,const Params&p){
 const c10::cuda::CUDAGuard guard(s.device());auto sizes=s.sizes().vec();sizes[0]=NP;auto out=at::empty(sizes,s.options());int64_t n=m.numel();
 props_kernel<<<blocks(n),256,0,c10::cuda::getCurrentCUDAStream()>>>(n,s.data_ptr<double>(),m.data_ptr<bool>(),t.data_ptr<double>(),out.data_ptr<double>(),p);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<Tensor> advance_cuda(const Tensor&s,const Tensor&m,const Tensor&a,const Tensor&tr,const Tensor&h,const Tensor&t,const Params&p){
 const c10::cuda::CUDAGuard guard(s.device());auto out=at::empty_like(s);auto sizes=s.sizes().vec();sizes[0]=ND;auto d=at::empty(sizes,s.options());int64_t n=m.numel();
 advance_kernel<<<blocks(n),256,0,c10::cuda::getCurrentCUDAStream()>>>(n,s.data_ptr<double>(),m.data_ptr<bool>(),a.data_ptr<bool>(),tr.data_ptr<double>(),h.data_ptr<double>(),t.data_ptr<double>(),out.data_ptr<double>(),d.data_ptr<double>(),p);C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,d};
}
Tensor contacts_cuda(const Tensor&ci,const Tensor&an,const Tensor&s,const Tensor&tr,const Tensor&v,const Tensor&a,const Params&p){
 const c10::cuda::CUDAGuard guard(s.device());int64_t n=s.numel()/NS,m=ci.numel();auto out=at::empty({NCONTACT,m},s.options());if(m==0)return out;
 contacts_kernel<<<blocks(m),256,0,c10::cuda::getCurrentCUDAStream()>>>(m,n,ci.data_ptr<int64_t>(),an.data_ptr<bool>(),s.data_ptr<double>(),tr.data_ptr<double>(),v.data_ptr<double>(),a.data_ptr<bool>(),out.data_ptr<double>(),p);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
Tensor current_cuda(const Tensor&s,const Tensor&tr,const Tensor&m,const Params&p){
 const c10::cuda::CUDAGuard guard(s.device());auto sizes=s.sizes().vec();sizes[0]=3;auto out=at::empty(sizes,s.options());int64_t n=m.numel();
 current_kernel<<<blocks(n),256,0,c10::cuda::getCurrentCUDAStream()>>>(n,s.data_ptr<double>(),tr.data_ptr<double>(),m.data_ptr<bool>(),out.data_ptr<double>(),p);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
Tensor stencil_cuda(const Tensor&c,const Tensor&x){
 const c10::cuda::CUDAGuard guard(x.device());auto out=at::empty_like(x);int64_t n=x.numel();
 stencil_kernel<<<blocks(n),256,0,c10::cuda::getCurrentCUDAStream()>>>(n,c.data_ptr<double>(),x.data_ptr<double>(),out.data_ptr<double>(),int(x.size(-1)));C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
