//
// Created by florian on 23.02.18.
//

#include "submesh.hpp"
#include <iostream>
#include <map>

void submesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F, std::set<int> &vertex_indices,
             Eigen::MatrixXd &SV, Eigen::MatrixXi &SF, std::map<int, int> &mvi2smvi, std::map<int, int> &mfi2smfi) {
    //std::map<int, int> mfi2smfi;
    //std::map<int, int> mvi2smvi;
    std::vector<Eigen::MatrixXi> submesh_faces;
    std::vector<Eigen::MatrixXd> submesh_vertices;
    int curr_face_index = 0;
    int curr_vertex_index = 0;
    for (int i = 0; i < F.rows(); i++) {
        if (std::find(vertex_indices.begin(), vertex_indices.end(), F(i, 0)) != vertex_indices.end() &&
            std::find(vertex_indices.begin(), vertex_indices.end(), F(i, 1)) != vertex_indices.end() &&
            std::find(vertex_indices.begin(), vertex_indices.end(), F(i, 2)) != vertex_indices.end()) {
            /// Import Face into submesh:
            /// 1. Check if all three vertices are already inside the submesh_vertíces set
            if (mvi2smvi.find(F(i, 0)) == mvi2smvi.end()) {
                mvi2smvi[F(i, 0)] = curr_vertex_index;
                submesh_vertices.push_back(V.row(F(i, 0)));
                curr_vertex_index++;
            }
            if (mvi2smvi.find(F(i, 1)) == mvi2smvi.end()) {
                mvi2smvi[F(i, 1)] = curr_vertex_index;
                submesh_vertices.push_back(V.row(F(i, 1)));
                curr_vertex_index++;
            }
            if (mvi2smvi.find(F(i, 2)) == mvi2smvi.end()) {
                mvi2smvi[F(i, 2)] = curr_vertex_index;
                submesh_vertices.push_back(V.row(F(i, 2)));
                curr_vertex_index++;
            }
            mfi2smfi[i] = curr_face_index;
            curr_face_index++;
            Eigen::MatrixXi tmp_face = Eigen::MatrixXi::Zero(1, 3);
            tmp_face(0, 0) = mvi2smvi[F(i, 0)];
            tmp_face(0, 1) = mvi2smvi[F(i, 1)];
            tmp_face(0, 2) = mvi2smvi[F(i, 2)];
            submesh_faces.push_back(tmp_face);
        }
    }
    SV = Eigen::MatrixXd::Zero(curr_vertex_index, 3);
    SF = Eigen::MatrixXi::Zero(curr_face_index, 3);
    /// Create Vertex Matrix from Vector:
    for (int i = 0; i < submesh_vertices.size(); i++) {
        SV.row(i) = submesh_vertices[i].row(0);
    }
    /// Create Face Matrix from Vector:
    for (int i = 0; i < submesh_faces.size(); i++) {
        SF.row(i) = submesh_faces[i];
    }
    std::cout << "Done!" << std::endl;
}