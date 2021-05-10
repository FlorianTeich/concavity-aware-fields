#ifndef FUNCTIONAL_OBJECT_UNDERSTANDING_SUBMESH_HPP
#define FUNCTIONAL_OBJECT_UNDERSTANDING_SUBMESH_HPP

#include <Eigen/Dense>
#include <vector>
#include <set>
#include <map>

/// Sample
void submesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F, std::set<int> &vertex_indices,
                    Eigen::MatrixXd &SV, Eigen::MatrixXi &SF, std::map<int, int> &mvi2smvi, std::map<int, int> &mfi2smfi);


#endif //FUNCTIONAL_OBJECT_UNDERSTANDING_SUBMESH_HPP
