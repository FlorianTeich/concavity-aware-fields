#include <stdlib.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/print.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/cpc_segmentation.h>
#include <Eigen/Dense>
#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>
#include <igl/edges.h>
#include <igl/adjacency_list.h>
#include <igl/avg_edge_length.h>
#include "merge.hpp"
#include "cpc.hpp"

typedef pcl::PointXYZRGBA PointT;  // The point type used for input

void cpc(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &label,
          float voxel_resolution,
          float seed_resolution,
          float color_importance,
          float spatial_importance,
          float normal_importance,
          bool use_single_cam_transform,
          bool use_supervoxel_refinement,
          float concavity_tolerance_threshold,
          float smoothness_threshold,
          uint32_t min_segment_size,
          bool use_extended_convexity,
          bool use_sanity_criterion,
          float min_cut_score,
          unsigned int max_cuts,
          unsigned int cutting_min_segments,
          bool use_local_constrain,
          bool use_directed_cutting,
          bool use_clean_cutting,
          unsigned int ransac_iterations,
          unsigned int k_factor,
          float min_part_size)
{
  pcl::PointCloud<PointT>::Ptr input_cloud_ptr (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr input_normals_ptr (new pcl::PointCloud<pcl::Normal>);
  pcl::PCLPointCloud2 input_pointcloud2;
  //pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  //pcl::console::setVerbosityLevel(pcl::console::L_WARN);

  ///  Default values of parameters before parsing
  if (use_extended_convexity)
    k_factor = 1;

  pcl::console::print_info ("Maximum cuts: %d\n", max_cuts);
  pcl::console::print_info ("Minimum segment size: %d\n", cutting_min_segments);
  pcl::console::print_info ("Use local constrain: %d\n", use_local_constrain);
  pcl::console::print_info ("Use directed weights: %d\n", use_directed_cutting);
  pcl::console::print_info ("Use clean cuts: %d\n", use_clean_cutting);
  pcl::console::print_info ("RANSAC iterations: %d\n", ransac_iterations);

  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

  std::vector<std::vector<int>> A;
  Eigen::MatrixXi E;
  Eigen::MatrixXd N;
  igl::per_vertex_normals(V, F, N);
  igl::edges(F, E);
  igl::adjacency_list(F, A);

  for (int i = 0; i < V.rows(); i++){
    pcl::Supervoxel<PointT> supervoxel_tmp;
    pcl::Normal normal_tmp;
    normal_tmp.normal_x = N(i,0);
    normal_tmp.normal_y = N(i,1);
    normal_tmp.normal_z = N(i,2);
    pcl::PointXYZRGBA centroid_tmp;
    centroid_tmp.x = V(i, 0);
    centroid_tmp.y = V(i, 1);
    centroid_tmp.z = V(i, 2);
    supervoxel_tmp.centroid_ = centroid_tmp;
    supervoxel_tmp.normal_ = normal_tmp;

    pcl::Supervoxel<PointT>::Ptr supervoxel_tmp_ptr = boost::make_shared<pcl::Supervoxel<PointT>>(supervoxel_tmp);

    supervoxel_clusters[i] = supervoxel_tmp_ptr;

  }

  std::multimap<uint32_t, uint32_t> supervoxel_adjacency;

  for (int i = 0; i < A.size(); i++){
    for (int j = 0; j < A[i].size(); j++){
      supervoxel_adjacency.insert(std::pair<uint32_t,uint32_t>(i, A[i][j]));
    }
  }

  /// Get the cloud of supervoxel centroid with normals and the colored cloud with supervoxel coloring (this is used for visulization)
  pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud =
          pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud (supervoxel_clusters);

  /// Set parameters for LCCP preprocessing and CPC (CPC inherits from LCCP, thus it includes LCCP's functionality)

  double avg_length = igl::avg_edge_length(V, F);

  PCL_INFO ("Starting Segmentation\n");
  pcl::CPCSegmentation<PointT> cpc;
  cpc.setConcavityToleranceThreshold (concavity_tolerance_threshold);
  cpc.setSanityCheck (use_sanity_criterion);
  cpc.setCutting (max_cuts, cutting_min_segments, min_cut_score, use_local_constrain, use_directed_cutting, use_clean_cutting);
  cpc.setRANSACIterations (ransac_iterations);
  cpc.setSmoothnessCheck (true, voxel_resolution, avg_length, smoothness_threshold);
  cpc.setKFactor (k_factor);
  cpc.setInputSupervoxels (supervoxel_clusters, supervoxel_adjacency);
  cpc.setMinSegmentSize (min_segment_size);
  cpc.segment ();

  PCL_INFO ("Interpolation voxel cloud -> input cloud and relabeling\n");

  pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud (new pcl::PointCloud<pcl::PointXYZL>);

  for (int i = 0; i < V.rows(); i++){
    pcl::PointXYZL p_tmp;
    p_tmp.x = V(i,0);
    p_tmp.y = V(i,1);
    p_tmp.z = V(i,2);
    p_tmp.label = i;
    sv_labeled_cloud->push_back(p_tmp);
  }

  pcl::PointCloud<pcl::PointXYZL>::Ptr cpc_labeled_cloud = sv_labeled_cloud->makeShared ();
  cpc.relabelCloud (*cpc_labeled_cloud);

  /// Merging step
  //merge(cpc_labeled_cloud, V, F);
  label = Eigen::MatrixXd::Zero(V.rows(), 1);

  for (int i = 0; i < cpc_labeled_cloud->size(); i++){
    label(i,0) = cpc_labeled_cloud->points[i].label;
  }

  mergeSegments(V, F, label, min_part_size);

  for (int i = 0; i < cpc_labeled_cloud->size(); i++){
    cpc_labeled_cloud->points[i].label = label(i,0);
  }
}