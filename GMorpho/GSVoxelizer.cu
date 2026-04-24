/*
	Copyright (c) 2015-2017 Telecom ParisTech (France).
	Authors: Stephane Calderon and Tamy Boubekeur.
	All rights reserved.

	This file is part of Broxy, the reference implementation for
	the paper:
		Bounding Proxies for Shape Approximation.
		Stephane Calderon and Tamy Boubekeur.
		ACM Transactions on Graphics (Proc. SIGGRAPH 2017),
		vol. 36, no. 5, art. 57, 2017.

	You can redistribute it and/or modify it under the terms of the GNU
	General Public License as published by the Free Software Foundation,
	either version 3 of the License, or (at your option) any later version.

	Licensees holding a valid commercial license may use this file in
	accordance with the commercial license agreement provided with the software.

	This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
	WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
*/

#include<GL/glew.h>

#include <cfloat>

#include <Common/OpenGL.h>
#include <Common/BoundingVolume.h>

#include "cuda_math.h"
#include "MarchingCubesMesher.h"

#include "GSVoxelizer.h"
#include "PythonScriptWrapper.h"
#include "npy.h"

using namespace std;
using namespace MorphoGraphics;

// region CUDA resource management & print()

template<typename T>
void GSVoxelizer::FreeGPUResource (T ** res) {
	if (*res != 0) {
		cudaFree (*res);
		*res = 0;
	}
}

void GSVoxelizer::CheckCUDAError () {
	cudaError_t err = cudaGetLastError ();
	if (err != cudaSuccess) {
		GSVoxelizer::print ("CUDA Error : " + std::string (cudaGetErrorString (err)));
		throw Voxelizer::Exception ("CUDA Error: " + std::string (cudaGetErrorString (err)));
	}
}

void GSVoxelizer::print (const std::string & msg) {
	std::cout << "[GSVoxelizer]: " << msg << std::endl;
}

// endregion

// region lifecycles

// note: it's just CUDA arrays null init and freeing

GSVoxelizer::GSVoxelizer () {}

GSVoxelizer::~GSVoxelizer () {}

// endregion

void GSVoxelizer::Load (const std::string& path_3dgs_folder) {
    path_3dgs_folder_ = path_3dgs_folder;
    auto means_ = extract_means(path_3dgs_folder);
    std::cout << "[GSVoxelizer] : " << "loaded " << means_.size() << " means" << std::endl;
    std::cout << "[GSVoxelizer] : " << "debug - mean[511] " << means_[511] << std::endl;

    const vector<Vec3f> & P = means_;
    og_bbox_.init (P[0]);
    for (const auto & i : P) {
        og_bbox_.extendTo(i);
    }
}

/**
 * initialize bbox_, res_, data_res_
 */
void GSVoxelizer::ComputeGridAttributes (int base_res, float margin) {
	bbox_ = AxisAlignedBoundingBox(og_bbox_);

    base_res_ = base_res;
	float thickness = max (fabs (bbox_.max()[0] - bbox_.min()[0]),
	                       max (fabs (bbox_.max()[1] - bbox_.min()[1]),
	                            fabs (bbox_.max()[2] - bbox_.min()[2]))
	                      ) / base_res_;
	bbox_.extend (2 * thickness + margin);
	cell_size_ = max (fabs (bbox_.max()[0] - bbox_.min()[0]),
	                  max (fabs (bbox_.max()[1] - bbox_.min()[1]),
	                       fabs (bbox_.max()[2] - bbox_.min()[2])
	                      )
	                 ) / base_res_;

	for (unsigned int i = 0; i < 3; i++) {
		res_[i] = base_res_;
	}

	for (unsigned int i = 0; i < 3; i++)
		data_res_[i] = (unsigned int) ceil ((bbox_.max()[i] - bbox_.min()[i]) / cell_size_);
}

static std::vector<Vec3f> build_coordinates(const AxisAlignedBoundingBox &bbox, int res, int num_bit_vox) {
    assert(num_bit_vox == 1);

    int res_x = res, res_y = res, res_z = res / ((32 / num_bit_vox));
    float bbox_len0 = fabs(bbox.max()[0] - bbox.min()[0]);
    float bbox_len1 = fabs(bbox.max()[1] - bbox.min()[1]);
    float bbox_len2 = fabs(bbox.max()[2] - bbox.min()[2]);
    float max_bbox_length = std::max(bbox_len0, std::max(bbox_len1, bbox_len2));

    std::vector<Vec3f> positions;
    positions.reserve(res_x * res_y * res_z * 32);
    float voxel_size = max_bbox_length / ((float)res);
    for (int i = 0; i < res_x; i++) {
        for (int j = 0; j < res_y; j++) {
            for (int k = 0; k < res_z; k++) {
                for (int s = 0; s < 32; s++) {
                    Vec3f position(((float) i) * voxel_size,
                                   ((float) j) * voxel_size,
                                   ((float) (32 * k + s)) * voxel_size
                    );
                    position = position + bbox.min()
                               + 0.5f * Vec3f(voxel_size, voxel_size, voxel_size);
                    positions.push_back(position);
                }
            }
        }
    }

    return positions;
}

static std::vector<unsigned char> filter_density_to_packed_grid(
        const std::vector<float>& densities,
        const AxisAlignedBoundingBox &bbox, int res, int num_bit_vox,
        float threshold
) {
    assert(num_bit_vox == 1);

    std::vector<unsigned char> pack_vox_grid;
    int pack_res_x = res / 2, pack_res_y = res / 2, pack_res_z = res / 2;
    pack_vox_grid.resize (pack_res_x * pack_res_y * pack_res_z, 0);

    int res_x = res, res_y = res, res_z = res / ((32 / num_bit_vox));
    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++) {
            for (int k = 0; k < res_z; k++) {
                for (int s = 0; s < 32; s++) {
                    auto density = densities[i * res_y * res_z * 32 + j * res_z * 32 + k * 32 + s];
                    if (density >= threshold) {
                        Vec3i position(i, j, 32 * k + s);
                        Vec3i pack_position(position[0] / 2,
                                            position[1] / 2,
                                            position[2] / 2);
                        unsigned char num_pack_shift = position[0] % 2
                                                       + 2 * (position[1] % 2)
                                                       + 4 * (position[2] % 2);
                        unsigned char pack_val = 1 << num_pack_shift;
                        unsigned int pack_idx = pack_position[0]
                                                + pack_res_x * pack_position[1]
                                                + pack_res_x * pack_res_y * pack_position[2];
                        pack_vox_grid[pack_idx] |= pack_val;
                    }
                }
            }
        }
    return pack_vox_grid;
}

void GSVoxelizer::VoxelizeConservativeTSDF (int base_res, float margin) {
	// Important note: an OpenGL context is needed
	// to call this function. We first flush the OpenGL
	// error pipeline.
	MorphoGraphics::GL::printOpenGLError ("Error before any OpenGL usage");

	// 1) Compute Grid Attributes
	ComputeGridAttributes (base_res, margin);

    int res = 2 * base_res_; // the base_res_ is the resolution of the packed layout
    int num_bit_vox = 1; // number of bits in one bucket

    std::cout << "[GSVoxelizer] : " << "start computing voxel grid" << std::endl;

	std::vector<unsigned char> pack_vox_grid = compute_packed_voxel_grid_from_tsdf(path_3dgs_folder_,bbox_, res, num_bit_vox);

	grid_.Init (bbox_, res_, data_res_, cell_size_);
	cudaMemcpy (grid_.grid_gpu ().voxels, &pack_vox_grid[0],
	            pack_vox_grid.size ()*sizeof (unsigned char),
	            cudaMemcpyHostToDevice);
	CheckCUDAError ();

	std::cout << "[GSVoxelizer] : " << "GridGPU structure set up done." << std::endl;
}

void GSVoxelizer::VoxelizeConservative (int base_res, float margin, float threshold, bool allow_cache) {
	// Important note: an OpenGL context is needed
	// to call this function. We first flush the OpenGL
	// error pipeline.
	MorphoGraphics::GL::printOpenGLError ("Error before any OpenGL usage");

	// This voxelization algorithm use the OpenGL pipeline to compute a
	// conservative voxelization of an input mesh. It use the conservative
	// rasterization capability of Maxwell and beyond NVIDIA GPUs.
	//
	// In this method, a voxel grid is encoded as an XY array of Z-Buckets.
	// G[X, Y] = Z-Buckets = [z-bucket[0], ... , z-bucket[N]]
	// And a z-bucket is an 'unsigned int' encoding 32 (resp. 4) z-voxels.
	// As such a voxel is encoded by 1 (resp. 8) bit(s).
	//
	// Important note: this voxel grid layout is not the same as the one
	// by CUDA GSVoxelizer (which is not conservative) and the morphological
	// operators. A subsequent step of conversion is needed to match
	// the specific packed layout of the CUDA GSVoxelizer and
	// the morphological operators.
	// A packed layout is made of a grid of unsigned char or bytes
	// that represent 8 packed voxels arranged in a 8 pieces cube.

	// 1) Compute Grid Attributes
	ComputeGridAttributes (base_res, margin);

    int res = 2 * base_res_; // the base_res_ is the resolution of the packed layout
    int num_bit_vox = 1; // number of bits in one bucket
    std::cout << "[GSVoxelizer] : " << "start computing densities" << std::endl;
    std::vector<float> densities;
    std::unique_ptr<std::vector<float>> cache = nullptr;
    if(allow_cache) {
        cache = cache_check_compute_densities(path_3dgs_folder_);
    }
    if(cache != nullptr) {
        // use cache
        densities = std::move(*cache);
    } else {
        // 2) compute coordinate for evaluation
        auto coordinates = build_coordinates(bbox_, res, num_bit_vox);
        // 3) calculate densities
        densities = compute_densities(std::move(coordinates), path_3dgs_folder_);
        cache_write_compute_densities(densities, path_3dgs_folder_);
    }
    std::cout << "[GSVoxelizer] : " << "density computed" << std::endl;

	// 4) Set up a GridGPU structure
    std::vector<unsigned char> pack_vox_grid = filter_density_to_packed_grid(
            densities, bbox_, res, num_bit_vox, threshold
    );
	grid_.Init (bbox_, res_, data_res_, cell_size_);

	cudaMemcpy (grid_.grid_gpu ().voxels, &pack_vox_grid[0],
	            pack_vox_grid.size ()*sizeof (unsigned char),
	            cudaMemcpyHostToDevice);
	CheckCUDAError ();
	std::cout << "[GSVoxelizer] : " << "GridGPU structure set up done." << std::endl;

	//TestByMeshing ("gl_vox_grid_mesh.off", grid_.grid_gpu ());

	//	ParseGrid (buck_vox_grid, bbox_, res, num_bit_vox, positions, normals);
	//	SaveCubeSet ("vox_grid_cubes.ply", positions, bbox_, res);
	//	std::cout << "[GSVoxelizer] : number of non empty cells : "
	//		<< positions.size () << std::endl;
}

__global__ 
void static ConvertVoxelGridByBitVoxelizer (unsigned char * voxelGrid, uint3 res, float * voxelGrid2x2x2) {
	unsigned int idX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idZ = blockIdx.z * blockDim.z + threadIdx.z;

	if (idX >= (2 * res.x - 1) || idY >= (2 * res.y - 1) || idZ >= (2 * res.z - 1))
		return;

	unsigned char val;
	val = voxelGrid[(idX / 2) + res.x * (idY / 2) + res.x * res.y * (idZ / 2)] & (1 << (idX % 2 + (idY % 2) * 2 + (idZ % 2) * 4));
	//	val = voxelGrid[(idX/2) + res.x*(idY/2) + res.x*res.y*(idZ/2)];

	voxelGrid2x2x2[idX + 2 * res.x * idY + 4 * res.x * res.y * idZ] = val ? 0.f : 1.f;
}

void GSVoxelizer::TestByMeshing (const std::string & filename,
                               const GridGPU & grid_gpu) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int num_cells_2x2x2 = 2 * grid_gpu.res.x * 2 * grid_gpu.res.y * 2 * grid_gpu.res.z;

	block_dim = dim3 (4, 4, 4);
	grid_dim = dim3 (((2 * grid_gpu.res.x) / block_dim.x) + 1,
	                 ((2 * grid_gpu.res.y) / block_dim.y) + 1,
	                 ((2 * grid_gpu.res.z) / block_dim.z) + 1);

	float * voxels_2x2x2 = NULL;
	cudaMalloc (&voxels_2x2x2, num_cells_2x2x2 * sizeof (float));
	CheckCUDAError ();
	std::cout << "[GSVoxelizer] : " << "TestByMeshing : voxels_2x2x2 allocated"
	          << " grid_gpu.res : "
	          << grid_gpu.res.x << ", " << grid_gpu.res.y << ", " << grid_gpu.res.z
	          << std::endl;

	time1 = GET_TIME ();
	ConvertVoxelGridByBitVoxelizer <<< grid_dim, block_dim>>>
	(grid_gpu.voxels,
	 grid_gpu.res,
	 voxels_2x2x2);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Convert Voxel Grid] "
	          << "convert by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	Vec3f bbox_min_mcm = Vec3f (grid_gpu.bbox_min.x,
	                            grid_gpu.bbox_min.y,
	                            grid_gpu.bbox_min.z);

	MarchingCubesMesher::Grid grid_mcm (bbox_min_mcm,
	                                    0.5f * grid_gpu.cell_size,
	                                    0.5f * grid_gpu.cell_size,
	                                    0.5f * grid_gpu.cell_size,
	                                    2 * grid_gpu.res.x,
	                                    2 * grid_gpu.res.y,
	                                    2 * grid_gpu.res.z);
	MarchingCubesMesher mesher (&grid_mcm);

	time1 = GET_TIME ();
	mesher.createMesh3D (voxels_2x2x2, 0.5f, 0.5 * NAN_EVAL);
	mesher.saveMesh (filename.c_str ());
	time2 = GET_TIME ();

	FreeGPUResource (&voxels_2x2x2);
}