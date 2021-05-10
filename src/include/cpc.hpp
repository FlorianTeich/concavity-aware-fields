//
// Created by florian on 23.02.18.
//

#ifndef FUNCTIONAL_OBJECT_UNDERSTANDING_CPC_LIB_HPP
#define FUNCTIONAL_OBJECT_UNDERSTANDING_CPC_LIB_HPP


void cpc(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &label,
         float voxel_resolution = 0.0075f,
         float seed_resolution = 0.03f,
         float color_importance = 0.0f,
         float spatial_importance = 1.0f,
         float normal_importance = 4.0f,
         bool use_single_cam_transform = false,
         bool use_supervoxel_refinement = false,
         float concavity_tolerance_threshold = 10,
         float smoothness_threshold = 0.1,
         uint32_t min_segment_size = 0,
         bool use_extended_convexity = false,
         bool use_sanity_criterion = false,
         float min_cut_score = 0.16,
         unsigned int max_cuts = 25,
         unsigned int cutting_min_segments = 400,
         bool use_local_constrain = false,
         bool use_directed_cutting = false,
         bool use_clean_cutting = false,
         unsigned int ransac_iterations = 10000,
         unsigned int k_factor = 0,
         float min_part_size = 0.02);

#endif //FUNCTIONAL_OBJECT_UNDERSTANDING_CPC_LIB_HPP
