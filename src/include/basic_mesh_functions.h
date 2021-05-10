#ifndef BASIC_MESH_FUNCTIONS_H
#define BASIC_MESH_FUNCTIONS_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <map>
#include <set>

typedef Eigen::Triplet<double> T;

void compute_vertex_normals(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &N);

void compute_opposite_vertices(Eigen::MatrixXi &E, Eigen::MatrixXi &F, Eigen::MatrixXi &OV);

void compute_incident_faces(Eigen::MatrixXi &E, Eigen::MatrixXi &F, Eigen::MatrixXi &IF);

void compute_face_normals(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &FC, Eigen::MatrixXd &FN);

void compute_dihedral_angle(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXi &IF,
                            Eigen::MatrixXi &OV, Eigen::MatrixXd &FN, Eigen::MatrixXd &DA);

void compute_distance(Eigen::MatrixXd &V, Eigen::MatrixXi &E, Eigen::MatrixXd &D);

void compute_distance(Eigen::MatrixXd &V, Eigen::MatrixXi &E, Eigen::MatrixXd &D);

int get_starting_point(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index,
                       std::vector<std::vector<int>> &A);

int get_starting_point_fast(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index,
                       std::vector<std::vector<int>> &A, Eigen::MatrixXi &E, std::vector<Eigen::Triplet<double>> &basic_tripletList);

int get_starting_point_fast(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index,
                            std::vector<std::vector<int>> &A, Eigen::MatrixXi &E, std::vector<Eigen::Triplet<double>> &basic_tripletList,
                            Eigen::MatrixXd &xx);

double max_geodesic_dist(std::set<int> &extreme_point_set, Eigen::MatrixXd &V);

bool in_proximity_to(Eigen::MatrixXd p1, std::set<int> &extreme_point_set, Eigen::MatrixXd &V, double dist_prox);

std::vector<int> get_extreme_points(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index_given,
                                    Eigen::MatrixXi &E);

void get_segmentation_field(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, Eigen::MatrixXi &E,
                            int index1, int index2, Eigen::MatrixXd &isoV, Eigen::MatrixXd &isoE,
                            Eigen::MatrixXd &z, std::vector<int> &isoF, std::vector<int> &isoI,
                            std::vector<Eigen::Triplet<double>> &basic_tripletList);

double getAngle3D (const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const bool in_degree);

void get_isoline_gradient_scores(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                        std::vector<std::vector<int>> &contour_faces, Eigen::MatrixXd &z, std::vector<double> &score);

void get_isoline_length(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                        std::vector<std::vector<int>> &contour_faces, std::vector<double> &length);

void add_mesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &V2, Eigen::MatrixXi &F2);

#endif