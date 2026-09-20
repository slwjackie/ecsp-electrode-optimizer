#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bc_engine.h"
namespace py=pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("run",&bc_native::run,py::call_guard<py::gil_scoped_release>(),"Compiled FP64 B/C integrator; masks [B,N,N], voltages [B].");
 m.attr("version")="8.2.1";
#if __cplusplus >= 202002L
 m.attr("compiler_standard")="c++20";
#else
 m.attr("compiler_standard")="c++17";
#endif
#ifdef WITH_CUDA
 m.attr("has_cuda")=true;
#else
 m.attr("has_cuda")=false;
#endif
}
