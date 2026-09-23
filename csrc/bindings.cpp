#include <pybind11/stl.h>
#include <torch/extension.h>

#include "hgemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V0 reference backends for LLM-TensorCore-HGEMM";
    m.def("hgemm", &hgemm, py::arg("a"), py::arg("b"),
          py::arg("implementation") = "auto", py::arg("epilogue") = "none");
    m.def("hgemm_out", &hgemm_out, py::arg("a"), py::arg("b"), py::arg("out"),
          py::arg("implementation"), py::arg("epilogue") = "none");
    m.def("has_cutlass", &has_cutlass);
    m.def("backend_info", &backend_info, py::arg("implementation"));
}
