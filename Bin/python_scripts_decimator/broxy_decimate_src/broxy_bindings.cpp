#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "Decimator.h"

namespace py = pybind11;

PYBIND11_MODULE(broxy_decimate, m) {
    m.doc() = "Broxy decimator bindings";

    py::class_<DecimationParams>(m, "DecimationParams")
        .def(py::init<>())
        .def_readwrite("use_linear_constraints", &DecimationParams::use_linear_constraints)
        .def_readwrite("use_features", &DecimationParams::use_features)
        .def_readwrite("global_scale", &DecimationParams::global_scale)
        .def_readwrite("target_error", &DecimationParams::target_error)
        .def_readwrite("target_num_faces", &DecimationParams::target_num_faces)
        .def_readwrite("target_num_vertices", &DecimationParams::target_num_vertices)
        .def_readwrite("max_edge_length_alpha", &DecimationParams::max_edge_length_alpha);

    py::class_<Decimator>(m, "Decimator")
        .def(py::init<const DecimationParams&, const Ref<const MatX3f>&, const Ref<const MatX3i>&>())
        .def(py::init<const DecimationParams&, const Ref<const MatX3f>&, const Ref<const MatX3i>&, const Ref<const MatX3f>&>())
        .def("is_done", &Decimator::is_done)
        .def("step", &Decimator::step)
        .def("step_until_vertex_threshold", &Decimator::step_until_vertex_threshold,
             py::arg("vertex_threshold"), py::arg("max_steps") = -1)
        .def("update_mesh", &Decimator::update_mesh)
        .def("get_current_vertices", &Decimator::get_current_vertices,
             py::return_value_policy::reference_internal)
        .def("get_current_faces", &Decimator::get_current_faces,
             py::return_value_policy::reference_internal)
        .def("get_current_normals", &Decimator::get_current_normals,
             py::return_value_policy::reference_internal)
        .def("get_current_error", &Decimator::get_current_error);
}
