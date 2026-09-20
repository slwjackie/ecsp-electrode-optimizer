#pragma once
#include "bc_ops.h"
#include <map>
#include <string>
namespace bc_native {
using Settings=std::map<std::string,double>;
std::map<std::string,Tensor> run(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Settings&);
}
