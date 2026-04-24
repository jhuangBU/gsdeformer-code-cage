#ifndef BROXY_MINIMAL_TYPES_H
#define BROXY_MINIMAL_TYPES_H


#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>

using namespace vcg;
using namespace tri;

/**********************************************************
    Mesh Classes for Quadric Edge collapse based simplification

    For edge collpases we need verteses with:
    - V->F adjacency
    - per vertex incremental mark
    - per vertex Normal


    Moreover for using a quadric based collapse the vertex class
    must have also a Quadric member Q();
    Otherwise the user have to provide an helper function object
    to recover the quadric.

 ******************************************************/
// The class prototypes.
class MyVertex;
class MyEdge;
class MyFace;

struct MyUsedTypes : public UsedTypes<Use<MyVertex>::AsVertexType, Use<MyEdge>::AsEdgeType, Use<MyFace>::AsFaceType> {
};

class MyVertex : public Vertex<MyUsedTypes,
            vertex::VFAdj,
            vertex::Coord3f,
            vertex::Normal3f,
            vertex::Mark,
            vertex::BitFlags> {
public:
    vcg::math::Quadric<double> &Qd() { return q; }
    bool is_feature() { return is_feature_; }
    void set_is_feature(bool is_feature) { is_feature_ = is_feature; }

private:
    math::Quadric<double> q;
    bool is_feature_;
};

class MyEdge : public Edge<MyUsedTypes> {
};

typedef BasicVertexPair<MyVertex> VertexPair;

class MyFace : public Face<MyUsedTypes,
            face::VFAdj,
            face::VertexRef,
            face::BitFlags> {
};

// the main mesh class
class MyMesh : public vcg::tri::TriMesh<std::vector<MyVertex>, std::vector<MyFace> > {
};

typedef typename MyMesh::ScalarType ScalarType;
typedef typename MyMesh::CoordType CoordType;
typedef MyMesh::VertexType::EdgeType EdgeType;
typedef typename MyMesh::VertexIterator VertexIterator;
typedef typename MyMesh::VertexPointer VertexPointer;
typedef typename MyMesh::FaceIterator FaceIterator;
typedef typename MyMesh::FacePointer FacePointer;


#endif //BROXY_MINIMAL_TYPES_H
