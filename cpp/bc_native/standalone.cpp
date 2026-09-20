// Standalone CPU executable. Links LibTorch/ATen, NOT libtorch_python.
// Transport protocol v1 is little-endian, bounded and versioned.
#include "bc_engine.h"
#include <ATen/Parallel.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <limits>
#include <type_traits>
namespace {
template<class T>T read(std::istream& in){T v;in.read(reinterpret_cast<char*>(&v),sizeof(v));if(!in)throw std::runtime_error("truncated native request");return v;}
template<class T>void write(std::ostream& out,T v){out.write(reinterpret_cast<const char*>(&v),sizeof(v));}
std::string text(std::istream&in){auto n=read<uint32_t>(in);if(n>4096)throw std::runtime_error("oversized key");std::string s(n,'\0');in.read(s.data(),n);if(!in)throw std::runtime_error("truncated key");return s;}
void text(std::ostream&out,const std::string&s){write<uint32_t>(out,s.size());out.write(s.data(),s.size());}
at::Tensor tensor(std::istream&in){
 auto type=read<uint8_t>(in);auto rank=read<uint8_t>(in);if(rank>8)throw std::runtime_error("invalid tensor rank");
 std::vector<int64_t> shape;uint64_t count=1;
 for(int i=0;i<rank;i++){auto d=read<int64_t>(in);if(d<0||d>1000000000LL||count>1000000000ULL/std::max<int64_t>(d,1))throw std::runtime_error("invalid tensor shape");shape.push_back(d);count*=d;}
 auto dtype=type==1?at::kDouble:type==2?at::kLong:at::kBool;
 if(type<1||type>3)throw std::runtime_error("unsupported tensor dtype");
 auto t=at::empty(shape,at::TensorOptions().dtype(dtype).device(at::kCPU));
 in.read(static_cast<char*>(t.data_ptr()),t.nbytes());if(!in)throw std::runtime_error("truncated tensor");return t;
}
void tensor(std::ostream&out,at::Tensor t){
 t=t.cpu().contiguous();auto type=t.scalar_type()==at::kDouble?1:t.scalar_type()==at::kLong?2:t.scalar_type()==at::kBool?3:0;
 if(!type)throw std::runtime_error("unsupported result dtype");write<uint8_t>(out,type);write<uint8_t>(out,t.dim());
 for(auto d:t.sizes())write<int64_t>(out,d);out.write(static_cast<const char*>(t.const_data_ptr()),t.nbytes());
}
}
int main(int argc,char**argv){
 if(argc==2&&std::string(argv[1])=="--version"){
#if __cplusplus >= 202002L
  constexpr const char* standard="C++20";
#else
  constexpr const char* standard="C++17";
#endif
  std::cout<<"ecsp_bc_native_cpu 8.2.1 ("<<standard<<" FP64 LibTorch; C++17-compatible core; no Python runtime)\n";return 0;
 }
 try{
  if(argc<3||argc>4)throw std::runtime_error("usage: ecsp_bc_native_cpu REQUEST.bin RESPONSE.bin [threads]");
  uint16_t endian=1;if(*reinterpret_cast<uint8_t*>(&endian)!=1)throw std::runtime_error("little-endian host required");
  int threads=argc==4?std::stoi(argv[3]):1;if(threads<1)throw std::runtime_error("threads must be positive");
  std::ifstream in(argv[1],std::ios::binary);if(!in)throw std::runtime_error("cannot open native request");
  char magic[8];in.read(magic,8);if(!in||std::memcmp(magic,"BCNATV1I",8))throw std::runtime_error("invalid request magic/version");
  auto n=read<uint32_t>(in);if(n>4096)throw std::runtime_error("oversized settings");bc_native::Settings cfg;
  for(uint32_t i=0;i<n;i++){auto name=text(in);auto v=read<double>(in);if(!std::isfinite(v))throw std::runtime_error("nonfinite setting");if(!cfg.emplace(name,v).second)throw std::runtime_error("duplicate setting");}
  auto a=tensor(in),c=tensor(in),v=tensor(in),table=tensor(in);if(in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing request bytes");
  // Validate the complete bounded request before initializing LibTorch's
  // thread pools.  This keeps malformed-input diagnostics reliable even when
  // the caller itself already hosts an OpenMP runtime.
  at::set_num_threads(threads);at::set_num_interop_threads(1);
  auto result=bc_native::run(a,c,v,table,cfg);
  std::ofstream out(argv[2],std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create native response");
  out.write("BCNATV1O",8);write<uint32_t>(out,result.size());for(auto&kv:result){text(out,kv.first);tensor(out,kv.second);}out.flush();if(!out)throw std::runtime_error("native response write failed");
  return 0;
 }catch(const std::exception&e){std::cerr<<"[bc-native-cpu] "<<e.what()<<'\n';return 2;}
}
