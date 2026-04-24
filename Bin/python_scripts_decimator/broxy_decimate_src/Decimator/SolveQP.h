//
// Created by johnb on 2026/1/23.
//

#ifndef BROXY_MINIMAL_SOLVEQP_H
#define BROXY_MINIMAL_SOLVEQP_H

#include "Types.h"

//-----------------------------------------------------------------------------
// Quadratic Programming :
// a) using FORTRAN routine
// b) defintion of a workspace utility class
//-----------------------------------------------------------------------------
#define MMAX 1000
#define NMAX 3
#define MNN (MMAX + NMAX + NMAX)
#define LWAR (3*NMAX*NMAX/2 + 10*NMAX + 2*MMAX + 1)
#define LIWAR NMAX

extern double c_qem_timing;
extern unsigned int c_qem_counter;

class QPWorkspace {
public:
    QPWorkspace() {
        qp_c_ = new double [NMAX * NMAX];
        qp_d_ = new double [NMAX];
        qp_a_ = new double [MMAX * NMAX];
        qp_b_ = new double [MMAX];
        qp_xl_ = new double [NMAX];
        qp_xu_ = new double [NMAX];
        qp_x_ = new double [NMAX];
        qp_u_ = new double [MNN];
        qp_war_ = new double [LWAR];
        qp_iwar_ = new int [LIWAR];
        qp_a_vec_ = new double [3 * MMAX];
        qp_b_vec_ = new double [MMAX];
    }

    ~QPWorkspace() {
        delete qp_c_;
        delete qp_d_;
        delete qp_a_;
        delete qp_b_;
        delete qp_xl_;
        delete qp_xu_;
        delete qp_x_;
        delete qp_u_;
        delete qp_war_;
        delete qp_iwar_;
        delete qp_a_vec_;
        delete qp_b_vec_;
    }

    inline double *qp_c() { return qp_c_; }
    inline double *qp_d() { return qp_d_; }
    inline double *qp_a() { return qp_a_; }
    inline double *qp_b() { return qp_b_; }
    inline double *qp_xl() { return qp_xl_; }
    inline double *qp_xu() { return qp_xu_; }
    inline double *qp_x() { return qp_x_; }
    inline double *qp_u() { return qp_u_; }
    inline double *qp_war() { return qp_war_; }
    inline int *qp_iwar() { return qp_iwar_; }
    inline double *qp_a_vec() { return qp_a_vec_; }
    inline double *qp_b_vec() { return qp_b_vec_; }

private:
    double *qp_c_;
    double *qp_d_;
    double *qp_a_;
    double *qp_b_;
    double *qp_xl_;
    double *qp_xu_;
    double *qp_x_;
    double *qp_u_;
    double *qp_war_;
    int *qp_iwar_;
    double *qp_a_vec_;
    double *qp_b_vec_;
};

void RunSolveQP(Eigen::Matrix3d &c, Eigen::Vector3d &d, Eigen::Matrix3d &omega, Eigen::Vector3d &sol,
                bool use_linear_constraints, bool &is_feasible, bool &is_optimal, bool &is_bounded, QPWorkspace *qp_ws,
                MyMesh::VertexType *v[2]);

#endif //BROXY_MINIMAL_SOLVEQP_H
