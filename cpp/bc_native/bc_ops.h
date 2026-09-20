#pragma once
#include <ATen/ATen.h>
#include <vector>
#include "bc_scalar.h"
namespace bc_native {
using at::Tensor;
Tensor props_cpu(const Tensor&,const Tensor&,const Tensor&,const Params&);
std::vector<Tensor> advance_cpu(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Params&);
Tensor contacts_cpu(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Params&);
Tensor current_cpu(const Tensor&,const Tensor&,const Tensor&,const Params&);
Tensor stencil_cpu(const Tensor&,const Tensor&);
#ifdef WITH_CUDA
Tensor props_cuda(const Tensor&,const Tensor&,const Tensor&,const Params&);
std::vector<Tensor> advance_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Params&);
Tensor contacts_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Params&);
Tensor current_cuda(const Tensor&,const Tensor&,const Tensor&,const Params&);
Tensor stencil_cuda(const Tensor&,const Tensor&);
#endif
inline Tensor properties(const Tensor&s,const Tensor&m,const Tensor&t,const Params&p){
#ifdef WITH_CUDA
 if(s.is_cuda())return props_cuda(s,m,t,p);
#endif
 return props_cpu(s,m,t,p);
}
inline std::vector<Tensor> advance(const Tensor&s,const Tensor&m,const Tensor&a,const Tensor&tr,const Tensor&h,const Tensor&t,const Params&p){
#ifdef WITH_CUDA
 if(s.is_cuda())return advance_cuda(s,m,a,tr,h,t,p);
#endif
 return advance_cpu(s,m,a,tr,h,t,p);
}
inline Tensor contacts(const Tensor&ci,const Tensor&an,const Tensor&s,const Tensor&tr,const Tensor&v,const Tensor&a,const Params&p){
#ifdef WITH_CUDA
 if(s.is_cuda())return contacts_cuda(ci,an,s,tr,v,a,p);
#endif
 return contacts_cpu(ci,an,s,tr,v,a,p);
}
inline Tensor currents(const Tensor&s,const Tensor&tr,const Tensor&m,const Params&p){
#ifdef WITH_CUDA
 if(s.is_cuda())return current_cuda(s,tr,m,p);
#endif
 return current_cpu(s,tr,m,p);
}
inline Tensor stencil(const Tensor&c,const Tensor&x){
#ifdef WITH_CUDA
 if(x.is_cuda())return stencil_cuda(c,x);
#endif
 return stencil_cpu(c,x);
}
}
