#include "Decimator.h"

void featureAwareDecimation (MorphoGraphics::Mesh& _cage, const MorphoGraphics::Mesh& _mesh, DecimationParams d) {
    auto vfp = from_morpho_mesh(_mesh);

    Decimator decimator(d, std::get<0>(vfp), std::get<1>(vfp), std::get<2>(vfp));

    while (true) {
        if(decimator.is_done()) {
            break;
        }
        decimator.step();
    }

    _cage = to_morpho_mesh(
        decimator.get_current_vertices(),
        decimator.get_current_faces(),
        decimator.get_current_normals()
    );

    std::cout << "Feature-Aware Decimator error : "
              << decimator.get_current_error()
              << " and number of faces : "
              << _cage.T ().size ()
                << " and number of vertices : "
                << _cage.P ().size ()
              << std::endl;
}

int main(int argc, char** argv) {
    MorphoGraphics::Mesh mesh;
    mesh.load("data/nerf_lego_dense_mesh.off");

    DecimationParams params;
    params.target_error = 5.0f;
    params.target_num_faces = 20;
    params.target_num_vertices = 150;
    params.max_edge_length_alpha = 2.5f;
    params.global_scale = 0.0629838f;

    MorphoGraphics::Mesh _cage;
    featureAwareDecimation(_cage, mesh, params);

    _cage.store("decimated_mesh.ply");

    return 0;
}
