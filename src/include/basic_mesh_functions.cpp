#include <iostream>
#include <random>
#include <chrono>
#include <boost/bind.hpp>
#include "basic_mesh_functions.h"
#include <igl/principal_curvature.h>
#include <igl/gaussian_curvature.h>
#include <igl/edges.h>
#include <igl/grad.h>
#include <igl/adjacency_list.h>
#include "custom_isoline.h"
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/SPQRSupport>
//#include <Eigen/SparseCholesky>
#include <chrono>

/**
 *
 * @param V
 * @param F
 * @param N
 */
void compute_vertex_normals(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &N) {
    N.resize(V.rows(), 3);
    int vertex_size = V.rows();
    std::vector<Eigen::Vector3d> *normal_buffer = new std::vector<Eigen::Vector3d>[vertex_size];

    /// Calculate Face Normals
    for (int i = 0; i < F.rows(); i++) {
        std::vector<Eigen::Vector3d> p(3);
        p[0] = V.row(F(i, 0));
        p[1] = V.row(F(i, 1));
        p[2] = V.row(F(i, 2));
        Eigen::Vector3d v1 = p[1] - p[0];
        Eigen::Vector3d v2 = p[2] - p[0];
        Eigen::Vector3d normal = v1.cross(v2);

        normal.normalize();
        normal_buffer[F(i, 0)].push_back(normal);
        normal_buffer[F(i, 1)].push_back(normal);
        normal_buffer[F(i, 2)].push_back(normal);
    }

    /// Now loop through each vertex vector, and avarage out all the normals stored.
    for (int i = 0; i < V.rows(); ++i) {
        for (int j = 0; j < normal_buffer[i].size(); ++j) {
            N(i, 0) += normal_buffer[i][j](0);
            N(i, 1) += normal_buffer[i][j](1);
            N(i, 2) += normal_buffer[i][j](2);
        }
        N(i, 0) /= normal_buffer[i].size();
        N(i, 1) /= normal_buffer[i].size();
        N(i, 2) /= normal_buffer[i].size();
    }
}

/**
 *
 * @param E
 * @param F
 * @param OV
 */
void compute_opposite_vertices(Eigen::MatrixXi &E, Eigen::MatrixXi &F, Eigen::MatrixXi &OV) {
    std::map<std::set<int>, std::vector<int>> edge_to_opposite_vertices_map;

    int not_assigned = F.rows();
    OV = Eigen::MatrixXi::Ones(E.rows(), 2);
    for (int i = 0; i < F.rows(); i++) {
        /// Register this face in all three edges if not already done
        edge_to_opposite_vertices_map[{F(i, 0), F(i, 1)}].push_back(F(i, 2));
        edge_to_opposite_vertices_map[{F(i, 1), F(i, 2)}].push_back(F(i, 0));
        edge_to_opposite_vertices_map[{F(i, 2), F(i, 0)}].push_back(F(i, 1));
    }

    for (int i = 0; i < E.rows(); i++) {
        OV(i, 0) = edge_to_opposite_vertices_map[{E(i, 0), E(i, 1)}][0];
        OV(i, 1) = edge_to_opposite_vertices_map[{E(i, 0), E(i, 1)}][1];
    }
}

/**
 *
 * @param E
 * @param F
 * @param IF
 */
void compute_incident_faces(Eigen::MatrixXi &E, Eigen::MatrixXi &F, Eigen::MatrixXi &IF) {
    std::map<std::set<int>, std::vector<int>> edge_to_faces_map;

    int not_assigned = F.rows();
    IF = Eigen::MatrixXi::Ones(E.rows(), 2);
    for (int i = 0; i < F.rows(); i++) {
        /// Register this face in all three edges if not already done
        edge_to_faces_map[{F(i, 0), F(i, 1)}].push_back(i);
        edge_to_faces_map[{F(i, 1), F(i, 2)}].push_back(i);
        edge_to_faces_map[{F(i, 2), F(i, 0)}].push_back(i);
    }

    for (int i = 0; i < E.rows(); i++) {
        IF(i, 0) = edge_to_faces_map[{E(i, 0), E(i, 1)}][0];
        IF(i, 1) = edge_to_faces_map[{E(i, 0), E(i, 1)}][1];
    }
}

/**
 *
 * @param V
 * @param F
 * @param FC
 * @param FN
 */
void compute_face_normals(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &FC, Eigen::MatrixXd &FN) {
    FC.resize(F.rows(), 3);
    FN.resize(F.rows(), 3);

    /// Calculate Face Normals
    for (int i = 0; i < F.rows(); i++) {
        std::vector<Eigen::Vector3d> p(3);
        p[0] = V.row(F(i, 0));
        p[1] = V.row(F(i, 1));
        p[2] = V.row(F(i, 2));
        Eigen::Vector3d v1 = p[1] - p[0];
        Eigen::Vector3d v2 = p[2] - p[0];
        Eigen::Vector3d normal = v1.cross(v2);

        normal.normalize();
        Eigen::Vector3d centroid = (p[0] + p[1] + p[2]) / 3;
        FC.row(i) = centroid;
        FN.row(i) = normal;
    }
}

/**
 *
 * @param V
 * @param F
 * @param E
 * @param IF
 * @param OV
 * @param FN
 * @param DA
 */
void compute_dihedral_angle(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXi &IF,
                            Eigen::MatrixXi &OV, Eigen::MatrixXd &FN, Eigen::MatrixXd &DA) {
    DA.resize(E.rows(), 1);
    std::map<std::set<int>, int> vertices_to_face_index_map;
    for (int i = 0; i < F.rows(); i++) {
        vertices_to_face_index_map[{F(i, 0), F(i, 1), F(i, 2)}] = i;
    }

    for (int i = 0; i < E.rows(); i++) {
        Eigen::Vector3d p0 = V.row(E(i, 0));
        Eigen::Vector3d p1 = V.row(E(i, 1));
        Eigen::Vector3d p2 = V.row(OV(i, 0));
        Eigen::Vector3d p3 = V.row(OV(i, 1));
        int f1 = vertices_to_face_index_map[{E(i, 0), E(i, 1), OV(i, 0)}];
        int f2 = vertices_to_face_index_map[{E(i, 0), E(i, 1), OV(i, 1)}];
        Eigen::Vector3d n1 = FN.row(f1);
        Eigen::Vector3d n2 = FN.row(f2);
        double normal_angle = atan2((n1.cross(n2)).norm(), n1.dot(n2));

        if ((p3 - p2).dot(n1 - n2) < 0)
            normal_angle = -normal_angle;

        DA(i) = normal_angle;
    }
}

/**
 *
 * @param V
 * @param E
 * @param D
 */
void compute_distance(Eigen::MatrixXd &V, Eigen::MatrixXi &E, Eigen::MatrixXd &D) {
    D.resize(E.rows(), 1);
    for (int i = 0; i < E.rows(); i++) {
        D(i) = sqrt((V(E(i, 0), 0) - V(E(i, 1), 0)) * (V(E(i, 0), 0) - V(E(i, 1), 0)) +
                    (V(E(i, 0), 1) - V(E(i, 1), 1)) * (V(E(i, 0), 1) - V(E(i, 1), 1)) +
                    (V(E(i, 0), 2) - V(E(i, 1), 2)) * (V(E(i, 0), 2) - V(E(i, 1), 2)));
    }
}

/**
 *
 * @param F
 * @param V
 * @param L
 * @param index
 * @param A
 * @return
 */
int get_starting_point(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index,
                       std::vector<std::vector<int>> &A) {

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(A[index].size() + 1, V.rows());
    for (int i = 0; i < A[index].size(); i++) {
        C(i, A[index][i]) = 1.0;
    }
    C(A[index].size(), index) = 1.0;

    Eigen::VectorXd B = Eigen::VectorXd::Ones(A[index].size() + 1);
    B(A[index].size(), 0) = 0.0;

    Eigen::VectorXd nulls = Eigen::VectorXd::Zero(V.rows());

    Eigen::MatrixXd MA(L.rows() + C.rows(), L.cols());
    MA << L,
            C;

    Eigen::VectorXd Mb(L.rows() + C.rows());
    Mb << nulls,
            B;

    Eigen::MatrixXd x = MA.householderQr().solve(Mb);
    Eigen::MatrixXd::Index maxRow, maxCol;
    x.maxCoeff(&maxRow, &maxCol);

    return maxRow;
}

/**
 *
 * @param F
 * @param V
 * @param L
 * @param index
 * @param A
 * @param E
 * @param basic_tripletList
 * @param x
 * @return
 */
int get_starting_point_fast(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index,
                            std::vector<std::vector<int>> &A, Eigen::MatrixXi &E,
                            std::vector<Eigen::Triplet<double>> &basic_tripletList,
                            Eigen::MatrixXd &x) {

    Eigen::SparseMatrix<double> LC_sparse(V.rows() + A[index].size() + 1, V.rows());
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList = basic_tripletList;
    tripletList.reserve(E.rows() * 2 + V.rows() + A[index].size());
    for (int i = 0; i < A[index].size(); i++) {
        tripletList.push_back(T(V.rows() + i, A[index][i], 1.0));
    }
    tripletList.push_back(T(V.rows() + A[index].size(), index, 1.0));
    LC_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(A[index].size() + 1, V.rows());
    for (int i = 0; i < A[index].size(); i++) {
        C(i, A[index][i]) = 1.0;
    }
    C(A[index].size(), index) = 1.0;

    Eigen::VectorXd B = Eigen::VectorXd::Ones(A[index].size() + 1);
    B(A[index].size(), 0) = 0.0;

    Eigen::VectorXd nulls = Eigen::VectorXd::Zero(V.rows());

    Eigen::VectorXd Mb(L.rows() + C.rows());
    Mb << nulls,
            B;

    auto start = std::chrono::steady_clock::now();
    Eigen::SPQR<Eigen::SparseMatrix<double>> solver;

    solver.compute(LC_sparse);

    x = solver.solve(Mb);

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in seconds : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " sec";

    Eigen::MatrixXd::Index maxRow, maxCol;
    x.maxCoeff(&maxRow, &maxCol);

    Eigen::MatrixXd::Index minRow, minCol;
    x.minCoeff(&minRow, &minCol);

    //std::cout << x << std::endl;

    //std::cout << ">> get_starting_point: from " << x(minRow,0) << " to " << x(maxRow,0) << std::endl;

    return maxRow;
}

/**
 * Obtain single point to start the search for extreme points from.
 *
 * @param F         Faces
 * @param V         Vertices
 * @param L         Laplacian graph matrix
 * @param index
 * @param A         Adjacency matrix
 * @param E         Edge list
 * @param basic_tripletList     triplet list
 * @return          index of vertex that is the resulting starting point
 */
int get_starting_point_fast(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index,
                            std::vector<std::vector<int>> &A, Eigen::MatrixXi &E,
                            std::vector<Eigen::Triplet<double>> &basic_tripletList) {

    Eigen::SparseMatrix<double> LC_sparse(V.rows() + A[index].size() + 1, V.rows());
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList = basic_tripletList;
    tripletList.reserve(E.rows() * 2 + V.rows() + A[index].size());
    for (int i = 0; i < A[index].size(); i++) {
        tripletList.push_back(T(V.rows() + i, A[index][i], 1.0));
    }
    tripletList.push_back(T(V.rows() + A[index].size(), index, 1.0));
    LC_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(A[index].size() + 1, V.rows());
    for (int i = 0; i < A[index].size(); i++) {
        C(i, A[index][i]) = 1.0;
    }
    C(A[index].size(), index) = 1.0;

    Eigen::VectorXd B = Eigen::VectorXd::Ones(A[index].size() + 1);
    B(A[index].size(), 0) = 0.0;

    Eigen::VectorXd nulls = Eigen::VectorXd::Zero(V.rows());

    Eigen::VectorXd Mb(L.rows() + C.rows());
    Mb << nulls,
            B;

    //Eigen::SPQR<Eigen::SparseMatrix<double>> solver;
    //solver.compute(LC_sparse);

    Eigen::SPQR<Eigen::SparseMatrix<double>> solver;

    solver.compute(LC_sparse);

    Eigen::MatrixXd x = solver.solve(Mb);

    Eigen::MatrixXd::Index maxRow, maxCol;
    x.maxCoeff(&maxRow, &maxCol);

    Eigen::MatrixXd::Index minRow, minCol;
    x.minCoeff(&minRow, &minCol);

    //std::cout << x << std::endl;

    //std::cout << ">> get_starting_point: from " << x(minRow,0) << " to " << x(maxRow,0) << std::endl;

    return maxRow;
}

/**
 *
 * @param extreme_point_set
 * @param V
 * @return
 */
double max_geodesic_dist(std::set<int> &extreme_point_set, Eigen::MatrixXd &V) {
    double dist_max = 0.0;
    for (auto ele1 : extreme_point_set) {
        for (auto ele2 : extreme_point_set) {
            if (ele1 < ele2)
                dist_max = std::max(dist_max, (V.row(ele1) - V.row(ele2)).norm());
        }
    }
    return dist_max;
}

/**
 *
 * @param p1
 * @param extreme_point_set
 * @param V
 * @param dist_prox
 * @return
 */
bool in_proximity_to(Eigen::MatrixXd p1, std::set<int> &extreme_point_set, Eigen::MatrixXd &V, double dist_prox) {
    bool is_prox = false;
    for (auto ele1 : extreme_point_set) {
        if ((V.row(ele1) - p1).norm() < dist_prox)
            is_prox = true;
    }
    return is_prox;
}

/**
 *
 * @param F
 * @param V
 * @param L
 * @param index_given
 * @param E
 * @return
 */
std::vector<int> get_extreme_points(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index_given,
                                    Eigen::MatrixXi &E) {

    std::vector<int> extreme_points;
    std::set<int> extreme_point_set;
    std::ranlux48 gen;
    std::uniform_int_distribution<int> uniform_0_255(0, V.rows() - 1);
    int prev_previous_num_points = 0;
    int previous_num_points = 0;
    int current_num_points = 0;
    int max_iters = 2;
    int iters = 0;

    std::vector<std::vector<int>> A;
    igl::adjacency_list(F, A);

    std::vector<T> basic_tripletList;
    basic_tripletList.reserve(E.rows() * 2 + V.rows() + 20);
    for (int i = 0; i < L.rows(); i++) {
        for (int j = 0; j < L.cols(); j++) {
            if (L(i, j) != 0.0)
                basic_tripletList.push_back(T(i, j, L(i, j)));
        }
    }

    //std::cout << ">> Finding extreme points...\n";
    do {

        iters++;
        int index = uniform_0_255(gen);
        std::cout << ">> Starting with random point " << index << "\n";

        int q = get_starting_point_fast(F, V, L, index, A, E, basic_tripletList);

        index = q;

        std::cout << ">> Selecting new point " << index << "\n";

        Eigen::SparseMatrix<double> LC_sparse(V.rows() + A[index].size() + 1, V.rows());

        std::vector<T> tripletList = basic_tripletList;

        for (int i = 0; i < A[index].size(); i++) {
            tripletList.push_back(T(V.rows() + i, A[index][i], 1.0));
        }
        tripletList.push_back(T(V.rows() + A[index].size(), index, 1.0));
        LC_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(A[index].size() + 1, V.rows());
        for (int i = 0; i < A[index].size(); i++) {
            C(i, A[index][i]) = 1.0;
        }
        C(A[index].size(), index) = 1.0;
        Eigen::VectorXd B = Eigen::VectorXd::Ones(A[index].size() + 1);
        B(A[index].size(), 0) = 0.0;
        Eigen::VectorXd nulls = Eigen::VectorXd::Zero(V.rows());

        Eigen::VectorXd Mb(L.rows() + C.rows());
        Mb << nulls,
                B;

        Eigen::SPQR<Eigen::SparseMatrix<double>> solver;
        solver.compute(LC_sparse);

        Eigen::MatrixXd x = solver.solve(Mb);

        Eigen::MatrixXd::Index maxRow, maxCol;
        x.maxCoeff(&maxRow, &maxCol);

        Eigen::MatrixXd::Index minRow, minCol;
        x.minCoeff(&minRow, &minCol);

        //std::cout << ">> Field varies from " << x(minRow,0) << " to " << x(maxRow,0) << std::endl;

        std::vector<int> candidate_extreme_points;

        std::vector<std::pair<double, int>> extreme_point_queue;

        for (int i = 0; i < V.rows(); i++) {
            double max_val = std::numeric_limits<double>::lowest();
            double min_val = std::numeric_limits<double>::max();
            for (int adj_vert : A[i]) {
                max_val = std::max(x(adj_vert, 0), max_val);
                min_val = std::min(x(adj_vert, 0), min_val);
            }
            if (x(i) >= max_val || x(i) <= min_val) {
                //std::cout << ">> Found extreme value: " << i << "\n";
                if (extreme_point_set.size() < 2) {
                    candidate_extreme_points.push_back(i);
                } else {
                    double max_dist = max_geodesic_dist(extreme_point_set, V);
                    if (!in_proximity_to(V.row(i), extreme_point_set, V, max_dist * 0.15)) {
                        double neighbors = 0.0;
                        for (int j = 0; j < A[i].size(); j++) {
                            neighbors = neighbors + x(A[i][j]);
                        }
                        neighbors = abs(x(i) - (neighbors / A[i].size()));
                        //std::cout << "Neighbor_values: " << neighbors << "\n";
                        extreme_point_queue.push_back(std::pair<double, int>(neighbors, i));
                    }
                }
            }
        }

        if (extreme_point_queue.size() > 0 && extreme_point_set.size() >= 2) {
            std::sort(extreme_point_queue.begin(), extreme_point_queue.end(),
                      boost::bind(&std::pair<double, int>::first, _1) <
                      boost::bind(&std::pair<double, int>::first, _2));

            for (int i = extreme_point_queue.size() - 1; i >= 0; i--) {
                double max_dist = max_geodesic_dist(extreme_point_set, V);
                if (!in_proximity_to(V.row(extreme_point_queue[i].second), extreme_point_set, V, max_dist * 0.15)) {
                    extreme_point_set.insert(extreme_point_queue[i].second);
                    //std::cout << ">> Added extreme point " << extreme_point_queue[i].second << std::endl;
                } else {
                    //std::cout << ">> Rejected extreme point " << extreme_point_queue[i].second << std::endl;
                }

            }
        }

        if (candidate_extreme_points.size() > 0) {
            std::vector<double> pair_dist;
            std::vector<int> candidate1;
            std::vector<int> candidate2;
            int selected_1 = 0;
            int selected_2 = 0;
            /// Add pair, that is the farthest apart first
            for (int i = 0; i < candidate_extreme_points.size(); i++) {
                for (int j = 0; j < i; j++) {
                    pair_dist.push_back(
                            (V.row(candidate_extreme_points[i]) - V.row(candidate_extreme_points[j])).norm());
                    candidate1.push_back(candidate_extreme_points[i]);
                    candidate2.push_back(candidate_extreme_points[j]);
                }
            }
            double max_dist = 0.0;
            for (int i = 0; i < pair_dist.size(); i++) {
                if (max_dist < pair_dist[i]) {
                    max_dist = pair_dist[i];
                    selected_1 = candidate1[i];
                    selected_2 = candidate2[i];
                }
            }
            extreme_point_set.insert(selected_1);
            extreme_point_set.insert(selected_2);
        }

    } while (iters < max_iters);

    extreme_points.clear();
    for (auto val : extreme_point_set) {
        //std::cout << ">> Final Extreme Point: " << val << std::endl;
        extreme_points.push_back(val);
    }
    return extreme_points;
}

/**
 *
 * @param F
 * @param V
 * @param L
 * @param E
 * @param index1
 * @param index2
 * @param isoV
 * @param isoE
 * @param z
 * @param isoF
 * @param isoI
 * @param basic_tripletList
 */
void get_segmentation_field(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, Eigen::MatrixXi &E,
                            int index1, int index2, Eigen::MatrixXd &isoV, Eigen::MatrixXd &isoE,
                            Eigen::MatrixXd &z, std::vector<int> &isoF, std::vector<int> &isoI,
                            std::vector<T> &basic_tripletList) {

    Eigen::SparseMatrix<double> LC_sparse(V.rows() + 2, V.rows());
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList = basic_tripletList;
    tripletList.reserve(E.rows() * 2 + V.rows() + 2);

    /*for (int iii = 0; iii < E.rows(); iii++){
        tripletList.push_back(T(E(iii,0), E(iii, 0), L(E(iii,0), E(iii, 0))));
        tripletList.push_back(T(E(iii,0), E(iii, 1), L(E(iii,0), E(iii, 1))));
        tripletList.push_back(T(E(iii,1), E(iii, 0), L(E(iii,1), E(iii, 0))));
    }*/

    //std::cout << "waiting...\n";
    /*for(int i = 0; i < L.rows(); i++)
    {
        for(int j = 0; j < L.cols(); j++){
            if (L(i,j) != 0.0)
                basic_tripletList.push_back(T(i,j,L(i,j)));
        }
    }*/
    //std::cout << "done...\n";

    tripletList.push_back(T(V.rows(), index1, 1.0));
    tripletList.push_back(T(V.rows() + 1, index2, 1.0));
    LC_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(2, V.rows());
    C(0, index1) = 1.0;
    C(1, index2) = 1.0;

    Eigen::VectorXd B = Eigen::VectorXd::Ones(2);
    B(0, 0) = 0.0;
    Eigen::VectorXd nulls = Eigen::VectorXd::Zero(V.rows());

    Eigen::VectorXd Mb(L.rows() + C.rows());
    Mb << nulls,
            B;

    Eigen::SPQR<Eigen::SparseMatrix<double>> solver;
    solver.compute(LC_sparse);

    z = solver.solve(Mb);

    //std::cout << ">> Quality of segmentation field: " << 1.0 / sqrt( abs(z.minCoeff()) + (1.0 + abs(z.maxCoeff() - 1.0))) << std::endl;

    isolines(V, F, z, 50, isoV, isoE, isoF, isoI);
}

/**
 *
 * @param isoV
 * @param isoE
 * @param contours
 * @param contour_faces
 * @param z
 * @param score
 */
void get_isoline_gradient_scores(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                                 std::vector<std::vector<int>> &contour_faces, Eigen::MatrixXd &z,
                                 std::vector<double> &score) {
    score.clear();
    for (int i = 0; i < contours.size(); i++) {
        double tmp_score = 0.0;
        double tmp_length = 0.0;
        int last_vertex = -1;
        for (int j = 0; j < contours[i].size(); j++) {
            /// Calculate Distance of current edge
            /// If first element: create edge to very last vertex
            if (j == 0)
                last_vertex = contours[i].back();
            /// Multiply edgelength by faces magnitude
            auto p1 = isoV.row(contours[i][j]);
            auto p2 = isoV.row(last_vertex);

            tmp_score = tmp_score + ((p1 - p2).norm() * z(contour_faces[i][j], 0));
            tmp_length = tmp_length + (p1 - p2).norm();

            last_vertex = contours[i][j];
        }
        score.push_back(tmp_score / tmp_length);
    }
}

/**
 *
 * @param isoV
 * @param isoE
 * @param contours
 * @param contour_faces
 * @param z
 * @param score
 */
void get_isoline_shape_scores(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                              std::vector<std::vector<int>> &contour_faces, Eigen::MatrixXd &z,
                              std::vector<double> &score) {
    score.clear();
    for (int i = 0; i < contours.size(); i++) {
        double tmp_score = 0.0;
        double tmp_length = 0.0;
        int last_vertex = -1;
        for (int j = 0; j < contours[i].size(); j++) {
            /// Calculate Distance of current edge
            /// If first element: create edge to very last vertex
            if (j == 0)
                last_vertex = contours[i].back();
            /// Multiply edgelength by faces magnitude
            auto p1 = isoV.row(contours[i][j]);
            auto p2 = isoV.row(last_vertex);

            tmp_score = tmp_score + ((p1 - p2).norm() * z(contour_faces[i][j], 0));
            tmp_length = tmp_length + (p1 - p2).norm();

            last_vertex = contours[i][j];
        }
        score.push_back(tmp_score / tmp_length);
    }
}

/**
 *
 * @param isoV
 * @param isoE
 * @param contours
 * @param contour_faces
 * @param length
 */
void get_isoline_length(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                        std::vector<std::vector<int>> &contour_faces, std::vector<double> &length) {
    length.clear();
    for (int i = 0; i < contours.size(); i++) {
        double tmp_length = 0.0;
        int last_vertex = -1;
        for (int j = 0; j < contours[i].size(); j++) {
            if (j == 0)
                last_vertex = contours[i].back();
            auto p1 = isoV.row(contours[i][j]);
            auto p2 = isoV.row(last_vertex);
            tmp_length = tmp_length + (p1 - p2).norm();
            last_vertex = contours[i][j];
        }
        length.push_back(tmp_length);
    }
}

/**
 *
 * @param V1
 * @param F1
 * @param V2
 * @param F2
 */
void add_mesh(Eigen::MatrixXd &V1, Eigen::MatrixXi &F1, Eigen::MatrixXd &V2, Eigen::MatrixXi &F2) {
    Eigen::MatrixXd V(V1.rows() + V2.rows(), V1.cols());
    V << V1, V2;
    Eigen::MatrixXi F(F1.rows() + F2.rows(), F1.cols());
    F << F1, (F2.array() + V1.rows());
    V1 = V;
    F1 = F;
}