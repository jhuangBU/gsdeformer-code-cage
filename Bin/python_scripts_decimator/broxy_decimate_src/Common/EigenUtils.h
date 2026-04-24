//
// Created by johnb on 2026/1/23.
//

#ifndef BROXY_MINIMAL_EIGENUTILS_H
#define BROXY_MINIMAL_EIGENUTILS_H

#include <Eigen/Core>
#include "Common/Mesh.h"

// row‑major so it matches NumPy C‑order (N,3)
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX3i = Eigen::Matrix<int,   Eigen::Dynamic, 3, Eigen::RowMajor>;
using Eigen::Ref;

static const MatX3f& empty_array() {
	static const MatX3f empty(0, 3);
	return empty;
}

static MorphoGraphics::Mesh to_morpho_mesh(
	const Ref<const MatX3f>& vertices,
	const Ref<const MatX3i>& faces,
	const Ref<const MatX3f>& normals
) {
	assert(vertices.rows() > 0);
	assert(vertices.cols() == 3);

	assert(faces.rows() > 0);
	assert(faces.cols() == 3);

	// normals: either empty or same row count as vertices, and 3 columns
	assert(normals.rows() == 0 || normals.rows() == vertices.rows());
	assert(normals.rows() == 0 || normals.cols() == 3);

	MorphoGraphics::Mesh m;
	m.clear();

	m.P().resize(vertices.rows());
	for (int i = 0; i < vertices.rows(); ++i) {
		m.P()[i] = MorphoGraphics::Vec3f(vertices(i, 0), vertices(i, 1), vertices(i, 2));
	}

	m.T().resize(faces.rows());
	for (int i = 0; i < faces.rows(); ++i) {
		m.T()[i] = MorphoGraphics::Mesh::Triangle(
			static_cast<unsigned int>(faces(i, 0)),
			static_cast<unsigned int>(faces(i, 1)),
			static_cast<unsigned int>(faces(i, 2))
		);
	}

	if (normals.rows() == 0) {
		m.recomputeNormals();
	} else {
		m.N().resize(normals.rows());
		for (int i = 0; i < normals.rows(); ++i) {
			m.N()[i] = MorphoGraphics::Vec3f(normals(i, 0), normals(i, 1), normals(i, 2));
		}
	}

	return m;
}

static void from_morpho_mesh_(
	const MorphoGraphics::Mesh& mesh,
	MatX3f& vertices,
	MatX3i& faces,
	MatX3f& normals
) {
	const auto& P = mesh.P();
	vertices.resize(static_cast<int>(P.size()), 3);
	for (int i = 0; i < static_cast<int>(P.size()); ++i) {
		vertices(i, 0) = P[i][0];
		vertices(i, 1) = P[i][1];
		vertices(i, 2) = P[i][2];
	}

	const auto& T = mesh.T();
	faces.resize(static_cast<int>(T.size()), 3);
	for (int i = 0; i < static_cast<int>(T.size()); ++i) {
		faces(i, 0) = static_cast<int>(T[i][0]);
		faces(i, 1) = static_cast<int>(T[i][1]);
		faces(i, 2) = static_cast<int>(T[i][2]);
	}

	const auto& N = mesh.N();
	if (!N.empty()) {
		normals.resize(static_cast<int>(N.size()), 3);
		for (int i = 0; i < static_cast<int>(N.size()); ++i) {
			normals(i, 0) = N[i][0];
			normals(i, 1) = N[i][1];
			normals(i, 2) = N[i][2];
		}
	} else {
		normals.resize(0, 3);
	}
}

static std::tuple<MatX3f, MatX3i, MatX3f> from_morpho_mesh(
	  const MorphoGraphics::Mesh& mesh
  ) {
	MatX3f vertices;
	MatX3i faces;
	MatX3f normals;
	from_morpho_mesh_(mesh, vertices, faces, normals);
	return {std::move(vertices), std::move(faces), std::move(normals)};
}

#endif //BROXY_MINIMAL_EIGENUTILS_H