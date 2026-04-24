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

#ifndef  GS_VOXELIZER_INC
#define  GS_VOXELIZER_INC

#include <string>

#include <Common/Vec3.h>
#include <Common/BoundingVolume.h>
#include <Common/Mesh.h>

#include "Grid.h"
#include "Voxelizer.h"

namespace MorphoGraphics {

class GSVoxelizer {
public:

    // lifecycles
	GSVoxelizer ();
	virtual ~GSVoxelizer ();

    // getters
	inline const AxisAlignedBoundingBox & bbox () const { return bbox_; }
	inline const Vec3ui & res () const { return res_; }
	inline const Vec3ui & data_res () const { return data_res_; }
	inline float cell_size () const { return cell_size_; }
	inline Grid grid () const { return grid_; }

    // compute
    // compute:IO
	void Load(const std::string& path_3dgs_folder); // checked: public
    // compute:voxelize
	void VoxelizeConservative (int base_res, float margin, float threshold = 1e-6, bool allow_cache=true); // checked: public
	void VoxelizeConservativeTSDF (int base_res, float margin); // checked: public
private:
    void ComputeGridAttributes (int base_res, float margin); // checked: private
protected:
	void print (const std::string & msg);
	void TestByMeshing (const std::string & filename, const GridGPU & grid_gpu);
private:
	// --------------------------------------------------------------
	//  CPU Data
	// --------------------------------------------------------------
	AxisAlignedBoundingBox bbox_;
    AxisAlignedBoundingBox og_bbox_;
	unsigned int base_res_;
	Vec3ui res_;
	Vec3ui data_res_;
	float cell_size_;

	// --------------------------------------------------------------
	//  OpenGL Data
	// --------------------------------------------------------------
    std::string path_3dgs_folder_;
	MorphoGraphics::Mesh mesh_;

	// --------------------------------------------------------------
	//  GPU Data
	// --------------------------------------------------------------
	Grid grid_;

	void CheckCUDAError ();
	template<typename T>
	void FreeGPUResource (T ** res);
};
}

#endif   /* ----- #ifndef GS_VOXELIZER_INC  ----- */
