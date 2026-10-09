#pragma once
#include "preflame_ops.h"
#include <map>
#include <string>
namespace preflame_cpp_cuda {
using Settings=std::map<std::string,double>;
std::map<std::string,Tensor> run(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Settings&);
}
