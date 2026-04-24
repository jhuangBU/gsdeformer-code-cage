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

#version 410
#extension GL_ARB_shader_image_load_store : require
#extension GL_ARB_shading_language_420pack : require

layout (binding = 0, r32ui) coherent uniform uimage3D vox_grid;

uniform uint res;

in PerVertex
{
  vec4 vGridSpacePosition;
} fs_data_in;

void main() {
	ivec3 final_grid_position;
 	ivec3 uGridSize = ivec3 (res); 
	vec3 clipSpacePosition = (fs_data_in.vGridSpacePosition.xyz + 1.0f) * 0.5f; // normalize to 0-1 range
	final_grid_position = ivec3(clipSpacePosition * uGridSize); // downcast for the indices

	// calculate val_vox, the mask to apply?
	uint val_vox = 0xffffffff;
	int num_shifts = 31 - (final_grid_position.z % 32);
	val_vox = (val_vox >> num_shifts);

	// final_grid_position to interbucket idx
	final_grid_position.z = final_grid_position.z/32;

	// 31 - intrabucket_idx
	// find the bucket, XOR (0xFFFF_FFFF >> 31 (0x0000_0001) for z=0, 0xFFFF_FFFF >> 31-3 (0x0000_000F) for z=1)
	// XOR: existing 0, set 0 -> 0
	// XOR: existing 0, set 1 -> 1
	// XOR: existing 1, set 0 -> 1
	// XOR: existing 1, set 1 -> 0
	imageAtomicXor (vox_grid, final_grid_position, val_vox);

	//  interbucket idx, z-1..0, flip for every single
	for (int i = final_grid_position.z - 1; i >= 0; i--) {
		final_grid_position.z = i;	
		val_vox = 0xffffffff;
		imageAtomicXor (vox_grid, final_grid_position, val_vox);
	}
}
