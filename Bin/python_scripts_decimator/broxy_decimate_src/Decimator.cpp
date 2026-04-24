#include "Decimator.h"
#include "Common/BoundingVolume.h"

using std::vector;
using MorphoGraphics::Vec3;
using MorphoGraphics::Vec3f;
using MorphoGraphics::Vec3ui;
using MorphoGraphics::Mesh;
using MorphoGraphics::AxisAlignedBoundingBox;
using std::max;

static const float MAX_SE_SIZE = 0.3f;
static const unsigned int VOXEL_RESOLUTION = 256;

static unsigned int base_res = VOXEL_RESOLUTION; // VOXEL RESOLUTION
static float se_size_ = MAX_SE_SIZE;
float margin = se_size_ + 2.f / (2 * ((float)base_res));

MockScaleField compute_mock_scale_field (const Mesh & mesh_, float global_scale) {
	// init AABB
	AxisAlignedBoundingBox bbox_;
	const vector<Vec3f> & P = mesh_.P();
	bbox_.init (P[0]);
	for (unsigned int i  = 0; i < P.size (); i++)
		bbox_.extendTo (P[i]);

	// compute base res and cell size & AABB with margin
	int base_res_ = base_res;
	float thickness = max (fabs (bbox_.max()[0] - bbox_.min()[0]),
						   max (fabs (bbox_.max()[1] - bbox_.min()[1]),
								fabs (bbox_.max()[2] - bbox_.min()[2]))
						  ) / base_res_;
	bbox_.extend (2 * thickness + margin);
	float cell_size_ = max (fabs (bbox_.max()[0] - bbox_.min()[0]),
					  max (fabs (bbox_.max()[1] - bbox_.min()[1]),
						   fabs (bbox_.max()[2] - bbox_.min()[2])
						  )
					 ) / base_res_;

	MorphoGraphics::Vec3<unsigned int> res_;
	for (unsigned int i = 0; i < 3; i++) {
		res_[i] = base_res_;
	}

	// build mock scale grid
	Vec3ui res = ((unsigned int)2) * res_;
	float cell_size = 0.5f * cell_size_;
	std::vector<char> scale_grid_vec (res[0]*res[1]*res[2], floor(global_scale/cell_size));

	return MockScaleField{bbox_, res, cell_size, scale_grid_vec};
}

Decimator::Decimator(
	const DecimationParams& params,
	const Ref<const MatX3f>& vertices,
	const Ref<const MatX3i>& faces
)
	: Decimator(params, vertices, faces, empty_array()) {}

Decimator::Decimator(
	const DecimationParams& params,
	const Ref<const MatX3f>& vertices,
	const Ref<const MatX3i>& faces,
	const Ref<const MatX3f>& normals
): params_(params),
heap_not_empty_(true), // so the first iteration runs
vertices(vertices), faces(faces), normals(normals) {
	auto input = to_morpho_mesh(vertices, faces, normals);

	// build scale grid data
	scale_field_ = compute_mock_scale_field (input, params.global_scale);
	Vec3f bbox_min = scale_field_.bbox.min ();
	Vec3f bbox_max = scale_field_.bbox.max ();
	float cell_size = scale_field_.cell_size;
	Vec3<unsigned int> res = scale_field_.res;
	auto scale_grid_cpu = scale_field_.scale_grid_vec.data();

	bool use_linear_constraints = params.use_linear_constraints;
	bool use_features = params.use_features;

	std::vector<bool> feature_taggs (input.P ().size (), false);

	float target_error = params.target_error;
	int target_num_faces = params.target_num_faces;
	float max_edge_length_alpha = params.max_edge_length_alpha;

	vcg_optim_session_ = std::make_unique<VCGOptimSession>(
		input,
		target_num_faces,
		target_error,
		max_edge_length_alpha,
		use_linear_constraints,
		use_features,
		bbox_min, bbox_max, res,
		cell_size, scale_grid_cpu,
		feature_taggs,
		3 // HC iterations
		);

}

// decimation loop //

bool Decimator::is_done() {
	bool has_more_faces = vcg_optim_session_->num_faces() > params_.target_num_faces;
	bool has_more_vertices = vcg_optim_session_->num_vertices() > params_.target_num_vertices;
	bool below_target_error = vcg_optim_session_->error() < params_.target_error;
	return !(heap_not_empty_ && (has_more_vertices && has_more_faces) && below_target_error);
}

void Decimator::step() {
	heap_not_empty_ = vcg_optim_session_->Optimize();
	Mesh cage_;
	vcg_optim_session_->GetMesh(cage_);
	from_morpho_mesh_(cage_, vertices, faces, normals);
}

int Decimator::step_until_vertex_threshold(int vertex_threshold, int max_steps) {
	int steps = 0;
	while (!is_done() && vcg_optim_session_->num_vertices() > vertex_threshold) {
		if (max_steps >= 0 && steps >= max_steps) {
			break;
		}
		heap_not_empty_ = vcg_optim_session_->Optimize();
		steps += 1;
	}

	Mesh cage_;
	vcg_optim_session_->GetMesh(cage_);
	from_morpho_mesh_(cage_, vertices, faces, normals);
	return steps;
}

void Decimator::update_mesh(Ref<const MatX3f> new_vertices) {
	assert(new_vertices.rows() == vertices.rows());
	assert(new_vertices.cols() == 3);

	vertices = new_vertices;
	normals.resize(0, 3);

	auto input = to_morpho_mesh(vertices, faces, empty_array());
	from_morpho_mesh_(input, vertices, faces, normals);

	assert(vcg_optim_session_->UpdateVertexPositions(input));

	heap_not_empty_ = true;
}
