#include "bc_ops.h"
#include <ATen/Parallel.h>
namespace bc_native {
Tensor props_cpu(const Tensor&s,const Tensor&m,const Tensor&t,const Params&p){
 int64_t n=m.numel();auto sizes=s.sizes().vec();sizes[0]=NP;auto out=at::empty(sizes,s.options());
 const double* sp=s.data_ptr<double>();const bool* mp=m.data_ptr<bool>();const double*tp=t.data_ptr<double>();double*o=out.data_ptr<double>();
 const Params* params=&p;
 at::parallel_for(0,n,1024,[n,sp,mp,tp,o,params](int64_t a,int64_t b){for(int64_t i=a;i<b;i++)props_cell(i,n,sp,mp,tp,o,*params);});return out;
}
std::vector<Tensor> advance_cpu(const Tensor&s,const Tensor&m,const Tensor&a,const Tensor&tr,const Tensor&h,const Tensor&t,const Params&p){
 int64_t n=m.numel();auto out=at::empty_like(s);auto sizes=s.sizes().vec();sizes[0]=ND;auto d=at::empty(sizes,s.options());
 const double*sp=s.data_ptr<double>();const bool*mp=m.data_ptr<bool>(),*ap=a.data_ptr<bool>();const double*rp=tr.data_ptr<double>(),*hp=h.data_ptr<double>(),*tp=t.data_ptr<double>();double*o=out.data_ptr<double>(),*dp=d.data_ptr<double>();
 const Params* params=&p;
 at::parallel_for(0,n,1024,[n,sp,mp,ap,rp,hp,tp,o,dp,params](int64_t x,int64_t y){for(int64_t i=x;i<y;i++)advance_cell(i,n,sp,mp,ap,rp,hp,tp,o,dp,*params);});return {out,d};
}
Tensor contacts_cpu(const Tensor&ci,const Tensor&an,const Tensor&s,const Tensor&tr,const Tensor&v,const Tensor&a,const Params&p){
 int64_t n=s.numel()/NS,m=ci.numel();auto out=at::empty({NCONTACT,m},s.options());
 TORCH_CHECK(n>0&&s.numel()==NS*n&&tr.numel()==NP*n,
             "BC native contact tensors have inconsistent state/property sizes");
 TORCH_CHECK(an.numel()==m&&a.numel()==p.batch&&v.numel()==p.batch&&
             n==int64_t(p.batch)*p.n*p.n,
             "BC native contact tensors have inconsistent contact/batch sizes");
 if(m==0)return out;
 const int64_t*cp=ci.data_ptr<int64_t>();const bool*ap=an.data_ptr<bool>(),*act=a.data_ptr<bool>();const double*sp=s.data_ptr<double>(),*tp=tr.data_ptr<double>(),*vp=v.data_ptr<double>();double*op=out.data_ptr<double>();
 // Capture every task operand explicitly by value.  Params remains alive for
 // the synchronous parallel_for call; a pointer keeps the hot task closure
 // small instead of copying the ~1 KiB immutable parameter block per task.
 const Params* params=&p;
 at::parallel_for(0,m,256,[cp,ap,sp,tp,vp,act,op,m,n,params](int64_t x,int64_t y){for(int64_t i=x;i<y;i++)contact_cell(i,m,n,cp,ap,sp,tp,vp,act,op,*params);});return out;
}
Tensor current_cpu(const Tensor&s,const Tensor&tr,const Tensor&m,const Params&p){
 int64_t n=m.numel();auto sizes=s.sizes().vec();sizes[0]=3;auto out=at::empty(sizes,s.options());
 const double*sp=s.data_ptr<double>(),*tp=tr.data_ptr<double>();const bool*mp=m.data_ptr<bool>();double*op=out.data_ptr<double>();
 const Params* params=&p;
 at::parallel_for(0,n,1024,[n,sp,tp,mp,op,params](int64_t a,int64_t b){for(int64_t i=a;i<b;i++)current_cell(i,n,sp,tp,mp,op,*params);});return out;
}
Tensor stencil_cpu(const Tensor&c,const Tensor&x){
 int64_t n=x.numel();auto out=at::empty_like(x);int side=int(x.size(-1));
 const double*cp=c.data_ptr<double>(),*xp=x.data_ptr<double>();double*yp=out.data_ptr<double>();
 at::parallel_for(0,n,2048,[n,cp,xp,yp,side](int64_t a,int64_t b){for(int64_t i=a;i<b;i++)stencil_cell(i,n,cp,xp,yp,side);});return out;
}
}
