#include "SolveQP.h"

#include <chrono>

static double get_time_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
#define GET_TIME get_time_ms

double c_qem_timing;
unsigned int c_qem_counter;

// the fortran routine
extern "C" int ql0001_(int *m, int *me, int *mmax, int *
                       n, int *nmax, int *mnn, double *c__, double *d__,
                       double *a, double *b, double *xl, double *xu,
                       double *x, double *u, int *iout, int *ifail, int *
                       iprint, double *war, int *lwar, int *iwar, int *liwar,
                       double *eps);

void RunSolveQP(Eigen::Matrix3d &c, Eigen::Vector3d &d, Eigen::Matrix3d &omega, Eigen::Vector3d &sol,
                bool use_linear_constraints, bool &is_feasible, bool &is_optimal, bool &is_bounded, QPWorkspace *qp_ws,
                MyMesh::VertexType *v[2]) {
    double time1, time2;
    time1 = GET_TIME();

    // Get the Quadratic Programming Workspace
    double *qp_a_vec = qp_ws->qp_a_vec();
    double *qp_b_vec = qp_ws->qp_b_vec();
    double *qp_c = qp_ws->qp_c();
    double *qp_d = qp_ws->qp_d();
    double *qp_a = qp_ws->qp_a();
    double *qp_b = qp_ws->qp_b();
    double *qp_xl = qp_ws->qp_xl();
    double *qp_xu = qp_ws->qp_xu();
    double *qp_x = qp_ws->qp_x();
    double *qp_u = qp_ws->qp_u();
    double *qp_war = qp_ws->qp_war();
    int *qp_iwar = qp_ws->qp_iwar();


    int curs_const = 0;
    // Constraints retrieval
    typename vcg::face::VFIterator<MyFace> fiter;
    for (fiter.F() = v[0]->VFp(), fiter.I() = v[0]->VFi(); fiter.F() != 0; ++fiter) {
        // for all faces in v0
        if (fiter.F()->V(0) != v[1] && fiter.F()->V(1) != v[1] && fiter.F()->V(2) != v[1]) {
            // skip faces with v1
            CoordType vcg_nn = NormalizedTriangleNormal(*fiter.F());
            CoordType vcg_p = fiter.F()->V(0)->P();
            Eigen::Vector3d nn(vcg_nn[0], vcg_nn[1], vcg_nn[2]);
            Eigen::Vector3d p(vcg_p[0], vcg_p[1], vcg_p[2]);
            Eigen::Vector3d t_nn(nn[0], nn[1], nn[2]);
            t_nn = omega * t_nn;
            qp_a_vec[3 * curs_const] = t_nn[0];
            qp_a_vec[3 * curs_const + 1] = t_nn[1];
            qp_a_vec[3 * curs_const + 2] = t_nn[2];
            qp_b_vec[curs_const] = -nn.dot(p);
            curs_const++;
        }
    }

    for (fiter.F() = v[1]->VFp(), fiter.I() = v[1]->VFi(); fiter.F() != 0; ++fiter) {
        // for all faces in v1
        CoordType vcg_nn = NormalizedTriangleNormal(*fiter.F());
        CoordType vcg_p = fiter.F()->V(0)->P();
        Eigen::Vector3d nn(vcg_nn[0], vcg_nn[1], vcg_nn[2]);
        Eigen::Vector3d p(vcg_p[0], vcg_p[1], vcg_p[2]);
        Eigen::Vector3d t_nn(nn[0], nn[1], nn[2]);
        t_nn = omega * t_nn;
        qp_a_vec[3 * curs_const] = t_nn[0];
        qp_a_vec[3 * curs_const + 1] = t_nn[1];
        qp_a_vec[3 * curs_const + 2] = t_nn[2];
        qp_b_vec[curs_const] = -nn.dot(p);
        curs_const++;
    }

    int m = use_linear_constraints ? curs_const : 0;
    int me = 0;
    int mmax = m;
    int n = 3;
    int nmax = n;
    int mnn = m + n + n;
    int iout, ifail, iprint = 0;
    int lwar = 3 * nmax * nmax / 2 + 10 * nmax + 2 * mmax + 1;
    int liwar = n;
    double eps = 1e-15;

    qp_iwar[0] = 1;

    // Objective Set Up
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // fortran77 by column storage
            qp_c[3 * j + i] = 2 * c(i, j);
        }
        qp_d[i] = d[i];
    }
    // Constraints Set Up
    for (int l = 0; l < m; l++) {
        for (int j = 0; j < 3; j++) {
            qp_a[m * j + l] = qp_a_vec[3 * l + j];
        }
        qp_b[l] = qp_b_vec[l];
    }
    for (int j = 0; j < 3; j++) {
        qp_xu[j] = 1e10;
        qp_xl[j] = -1e10;
    }

    ql0001_(&m, &me, &mmax, &n, &nmax, &mnn,
            qp_c, qp_d, qp_a, qp_b, qp_xl, qp_xu, qp_x, qp_u,
            &iout, &ifail, &iprint, qp_war, &lwar, qp_iwar, &liwar, &eps);

    is_feasible = false;
    is_optimal = false;
    is_bounded = false;

    if (ifail == 0) {
        // All good
        is_feasible = true;
        is_optimal = true;
        is_bounded = true;
        for (int j = 0; j < 3; j++)
            sol[j] = qp_x[j];
    } else if (ifail == 5) {
        // Lenght of working array too short
        std::cout << "Siconos : length of working array is too short" << std::endl;
    }

    time2 = GET_TIME();
    c_qem_timing += (time2 - time1);
    c_qem_counter++;
}
