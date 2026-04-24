#ifndef GMORPHO_PYTHONSCRIPTWRAPPER_H
#define GMORPHO_PYTHONSCRIPTWRAPPER_H

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <Common/BoundingVolume.h>

#include "Common/Vec3.h"

void invoke_python_script(const std::string& path_to_script,
                          const std::vector<std::string>& arguments,
                          const std::string& conda_env = "broxy-python-scripts");

std::vector<MorphoGraphics::Vec3f> extract_means(const std::string& path_3dgs_folder);

std::unique_ptr<std::vector<float>> cache_check_compute_densities(const std::string & path_3dgs_folder);

void cache_write_compute_densities(const std::vector<float>& densities, const std::string & path_3dgs_folder);

std::vector<float> compute_densities(
        std::vector<MorphoGraphics::Vec3f>&& coordinates,
        const std::string & path_3dgs_folder
);

std::vector<unsigned char> compute_packed_voxel_grid_from_tsdf(const std::string & path_3dgs_folder_, MorphoGraphics::AxisAlignedBoundingBox bbox, int res, int num_bit_vox);

std::string locate_3dgs_ply(const std::string & path_3dgs_folder, int iteration = -1);

#endif //GMORPHO_PYTHONSCRIPTWRAPPER_H
