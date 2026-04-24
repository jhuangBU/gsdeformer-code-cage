//
// Created by johnb on 2026/1/23.
//

#ifndef BROXY_MINIMAL_DECIMATOR_H
#define BROXY_MINIMAL_DECIMATOR_H

#include <memory>
#include <vector>
#include "Common/EigenUtils.h"
#include "Common/BoundingVolume.h"
#include "Decimator/Decimator.h"

struct MockScaleField {
	MorphoGraphics::AxisAlignedBoundingBox bbox;
	MorphoGraphics::Vec3ui res;
	float cell_size;
	std::vector<char> scale_grid_vec;
};

struct DecimationParams {
	bool use_linear_constraints = true;
	bool use_features = false;
	float global_scale;
	float target_error = 5.0f;
	int target_num_faces = 20;
	int target_num_vertices = 150;
	float max_edge_length_alpha = 1.0f;
};

/**
 * central Decimator object for python integration, provides eigen API for python interaction
 */
class Decimator {
public:

	// lifecycle //

	Decimator(
		const DecimationParams& params,
		const Ref<const MatX3f>& vertices,
		const Ref<const MatX3i>& faces
	);

	Decimator(
		const DecimationParams& params,
		const Ref<const MatX3f>& vertices,
		const Ref<const MatX3i>& faces,
		const Ref<const MatX3f>& normals
	);

	// decimation loop //

	bool is_done();
	void step();
	int step_until_vertex_threshold(int vertex_threshold, int max_steps = -1);

	// decimation modifier - changes the mesh during optimization //

	void update_mesh(Ref<const MatX3f> vertices);

	// decimation accessor //

	const MatX3f& get_current_vertices() const { return vertices; }

	const MatX3i& get_current_faces() const { return faces; }

	const MatX3f& get_current_normals() const { return normals; }

	float get_current_error() const { return vcg_optim_session_->error(); }

private:
	// config
	const DecimationParams params_;
	// optimization state
	bool heap_not_empty_;
	std::unique_ptr<VCGOptimSession> vcg_optim_session_;
	// scale grid data - must outlive vcg_optim_session_
	MockScaleField scale_field_;
	// aux optim state - synced to VCGOptimSession
	MatX3f vertices;
	MatX3i faces;
	MatX3f normals;
};

#endif //BROXY_MINIMAL_DECIMATOR_H