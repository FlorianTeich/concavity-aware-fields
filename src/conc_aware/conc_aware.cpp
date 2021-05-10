#include <iostream>
#include <random>
#include <memory>
#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/read_triangle_mesh.h>
#include <igl/edges.h>
#include <igl/principal_curvature.h>
#include <igl/gaussian_curvature.h>
#include <igl/grad.h>
#include <igl/decimate.h>
#include <igl/unique.h>
#include <igl/colormap.h>
#include <igl/writeOFF.h>
#include <igl/is_vertex_manifold.h>

//#include <igl/embree/reorient_facets_raycast.h>

#include <igl/opengl/glfw/background_window.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "custom_isoline.h"
#include "basic_mesh_functions.h"
#include "get_separate_lines.h"
#include "split_mesh.h"
#include "create_laplacian.h"
#include "color_ext.h"
#include "submesh.hpp"
#include "cpc.hpp"
#include "merge.hpp"
#include "meshseg_helpers.hpp"
//#include "reorient.cpp"

#include <pcl/io/pcd_io.h>

#ifdef IGL_STATIC_LIBRARY
  #include <igl/writeOFF.cpp>
  template bool igl::writeOFF<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&);
#endif

enum vis_mode{ vis_standard, vis_field, vis_field_gradient, vis_concavity, vis_weight};

enum isoline_vis_mode{ vis_none, vis_local_gs, vis_svs, vis_score, vis_id, vis_gs};

namespace po = boost::program_options;

static void ShowHelpMarker(const char* desc)
{
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered())
  {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

class CustomMenu : public igl::opengl::glfw::imgui::ImGuiMenu
{

public:
  std::vector<std::string> field_choices;
  std::vector<std::string> visualization_choices;
  std::vector<std::string> label_choices;
  float beta;
  float epsilon;
  float sigma;
  int field_choice;
  int visualization_choice;
  int label_choice;
  bool redraw;
  bool run_segmentation;
  bool segmentation_finished;
  bool use_cpc;
  bool draw_iso;
  std::vector<double> performance;

  virtual void draw_custom_window() override
  {
    // Define next window position + size
    ImGui::SetNextWindowPos(ImVec2(180.f * menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 440), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Laplacian Segmentation", nullptr, ImGuiWindowFlags_NoSavedSettings);

    ///GUI:
    // NumericalInput
    ImGui::InputFloat("Beta", &beta, 0.0f, 1.0f);
    ImGui::SameLine(); ShowHelpMarker("Beta is a weighting coefficient for concave connections used for creating the "
                                        "Graph Laplacian weight Matrix. The smaller Beta, the less will concave regions "
                                        "allow propagation. Thus, a higher Beta value will lead to more uniformly "
                                        "sampled isolines, whereas a low Beta value will result in isolines clustered "
                                        "at concave regions.\n");

    ImGui::InputFloat("Epsilon", &epsilon, 0.0f, 1.0f);
    ImGui::SameLine(); ShowHelpMarker("Small constant to avoid zero division.\n");

    ImGui::InputFloat("Sigma", &sigma, 0.0f, 1.0f);
    ImGui::SameLine(); ShowHelpMarker("Threshold value to decide whether connection of two vertices is convex or "
                                        "concave, based on the normal difference.\n");

    ImGui::Checkbox("Use CPC post-segmentation", &use_cpc);

    ImGui::Checkbox("Draw Isolines", &draw_iso);

    // Combobox Field Selection
    if (ImGui::Combo("Field", &field_choice, field_choices)) {
      redraw = true;
    };

    // Combobox Visualization Selection
    if (ImGui::Combo("Visualization", &visualization_choice, visualization_choices)) {
      redraw = true;
    };

    // Combobox Label Selection
    if (ImGui::Combo("Label", &label_choice, label_choices)) {
      redraw = true;
    };

    // Button "Segment"
    if (ImGui::Button("Segment")){
      std::cout << ">> Button Clicked\n";
      run_segmentation = true;
      segmentation_finished = false;
      redraw = true;
    }

    if (performance.size() > 0){
      ImGui::LabelText(std::to_string(performance[0]).c_str(), "HD");
      ImGui::LabelText(std::to_string(performance[1]).c_str(), "HD m");
      ImGui::LabelText(std::to_string(performance[2]).c_str(), "HD f");
      ImGui::LabelText(std::to_string(performance[3]).c_str(), "CE");
      ImGui::LabelText(std::to_string(performance[4]).c_str(), "CE");
      ImGui::LabelText(std::to_string(performance[5]).c_str(), "GCE");
      ImGui::LabelText(std::to_string(performance[6]).c_str(), "LCE");
      ImGui::LabelText(std::to_string(performance[7]).c_str(), "RI");
      ImGui::LabelText(std::to_string(performance[8]).c_str(), "CD");
    }

    ImGui::End();
  }

};

int main (int argc, char ** argv){

  /// Initialization
  //using namespace nanogui;
  int index = 0;
  std::string mesh_file = "";
  Eigen::MatrixXd V;    // Vertices
  Eigen::MatrixXi F;    // Faces
  Eigen::MatrixXi E;    // Edges
  Eigen::MatrixXd N;    // Normals
  std::vector<std::vector<int>> VF;
  std::vector<std::vector<int>> VFi;
  Eigen::MatrixXd dblA; // Area
  Eigen::MatrixXi IF;   // Incident Faces
  Eigen::MatrixXi OV;   // Opposite Vertices
  Eigen::MatrixXd FC;   //
  Eigen::MatrixXd FN;   // Face Normals
  Eigen::MatrixXd DA;   // Dihedral Angle
  Eigen::MatrixXd D;    // Distance
  Eigen::MatrixXd HL;   // Harmonic Laplacian
  Eigen::MatrixXd L;    // Laplacian
  Eigen::MatrixXd G;    // Gaussian Curvature
  Eigen::MatrixXd P1, P2; // Isolines
  std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> isoline_lines;
  std::vector<Eigen::MatrixXd> isoline_lines_id;
  std::vector<Eigen::MatrixXd> isoline_lines_color;
  Eigen::MatrixXd isoline_colors; // Isoline Colors
  Eigen::MatrixXd g_hat;
  std::vector<int> applied_isolines;
  std::vector<std::vector<double>> isoline_neighbor_length;
  std::vector<std::vector<Eigen::MatrixXd>> isoline_vertices;
  std::vector<std::vector<int>> isoline_face_ids;
  std::vector<double> isoline_length;
  std::vector<double> isoline_svs;
  std::vector<double> isoline_local_gs;
  std::vector<double> isoline_gs;
  std::vector<double> isoline_score;
  std::vector<int> isoline_field_id;
  std::vector<int> candidate_isoline_id;
  std::vector<std::vector<double>> candidate_neighbor_length;
  std::vector<std::vector<Eigen::MatrixXd>> candidate_vertices;
  std::vector<std::vector<int>> candidate_face_ids;
  std::vector<double> candidate_length;
  std::vector<double> candidate_svs;
  std::vector<double> candidate_gs;
  std::vector<double> candidate_score;
  std::vector<int> candidate_field_id;
  std::vector<int> extreme_points;
  int num_fields;
  std::vector<std::set<int>> isoline_face_set;
  std::vector<std::vector<int>> field_to_isoline_ids;
  Eigen::MatrixXd vertex_is_concave;
  bool redraw = false;
  bool draw_isoline = false;
  vis_mode mode = vis_standard;
  isoline_vis_mode isoline_mode = vis_none;
  std::vector<Eigen::MatrixXd> gradient_magnitude;
  std::vector<Eigen::MatrixXd> fields;
  Eigen::MatrixXd C;
  std::vector<int> extreme_p1;
  std::vector<int> extreme_p2;
  bool indicate_candidates = false;
  std::vector<std::vector<int>> edge_indices;
  bool show_candidates = true;
  bool show_local_gs_score = true;
  double beta, eps, sigma;
  float min_part_size;
  int clip_bound;
  Eigen::MatrixXd vertex_labels;
  Eigen::MatrixXd mesh_label_colored;
  bool use_cpc = false;
  std::vector<double> res;

  /// Parse Command Line Arguments

  po::options_description desc("MainOptions");
  desc.add_options()
    ("help,h", "Print help messages")
    ("nogui", "Suppress GUI")
    ("file,f",
     po::value<std::string>()->required(),
     "Input Mesh File, can be of formats OFF, PLY, STL or OBJ. Needs to be manifold")
    ("beta,b",
     po::value<double>()->default_value(0.1),
     "Concave weight factor")
    ("eps,e",
     po::value<double>()->default_value(0.000001),
     "Small constant to prevent zero division")
    ("sigma,s",
     po::value<double>()->default_value(0.001),
     "Concavity tolerance")
    ("minpartsize,m",
     po::value<float>()->default_value(0.02),
     "Minimal Part Size for CPC")
    ("use_cpc,c", po::bool_switch(&use_cpc), "description")
    ("out,o",
    po::value<std::string>()->required(),
    "Output file path");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  }
  catch (po::error &e) {
    /* Invalid options */
    std::cerr << "Error: " << e.what() << std::endl << std::endl;
    std::cout << "Boost program_options tester:" << std::endl
              << desc << std::endl;
    return 0;
  }
  try{
      po::notify(vm);
  }catch(std::exception& e)
  {
      std::cerr << "Error: " << e.what() << "\n";
      return false;
  }

  if (vm.count("help"))
  {
    /* print usage */
    std::cout << "Boost program_options tester:" << std::endl
              << desc << std::endl;
    return 0;
  }
  mesh_file = vm["file"].as<std::string>();
  std::string out = vm["out"].as<std::string>();
  beta = vm["beta"].as<double>();
  eps = vm["eps"].as<double>();
  sigma = vm["sigma"].as<double>();
  min_part_size = vm["minpartsize"].as<float>();

  /// Get Vertices & Faces
  if (boost::filesystem::is_regular_file(mesh_file)) {
      igl::read_triangle_mesh(mesh_file, V, F);
  }else{
      std::cerr << "File does not exist!" << std::endl;
      return -1;
  }

  Eigen::MatrixXi B;
  igl::is_vertex_manifold(F, B);
  if (B.minCoeff() == 0){
    std::cerr << ">> The loaded mesh is not manifold.\n";
  }

  auto f = [&](){
    int lap_weighting = 0;
    auto start = std::chrono::steady_clock::now();

    /// Compute Laplacian and Features
    //std::cout << ">> Computing all Features...\n";

    compute_all_features(V, F, E, N, VF, VFi, IF, OV, FC, FN, DA, D, L, G, dblA);

      auto end = std::chrono::steady_clock::now();
      std::cout << "COMPUTED FEATURES. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " msec\n";

    std::cout << ">> Computing Laplacian... e=" << eps << std::endl;
    compute_laplacian(V, F, E, G, N, L, vertex_is_concave, beta, eps, sigma, clip_bound, lap_weighting, 0.4);

      end = std::chrono::steady_clock::now();
      std::cout << "COMPUTED LAPLACIAN. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " msec\n";
    //Eigen::MatrixXd HL;
    //compute_laplacian_harmonic(V, F, E, HL, N, beta, sigma);
    //std::cout << ">> Finding extreme Points...\n";
    //compute_laplacian_harmonic(V, F, HL);
    //extreme_points = get_extreme_points(F, V, HL, index, E);
    extreme_points = get_extreme_points(F, V, L, index, E);

      end = std::chrono::steady_clock::now();
      std::cout << "EXTREME POINTS. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " msec\n";

    /// Visualize Extreme Points
    redraw = true;

    /// Main Routine: Compute Segmentation Fields and choose Isolines as candidates
    std::vector<T> basic_tripletList;
    basic_tripletList.reserve(E.rows() * 2 + V.rows() + 20);
    for(int i = 0; i < L.rows(); i++)
    {
      for(int j = 0; j < L.cols(); j++){
        if (L(i,j) != 0.0)
          basic_tripletList.push_back(T(i,j,L(i,j)));
      }
    }

    num_fields = 0;
    int num_isos = 0;
    for (int point1 : extreme_points) {
      for (int point2 : extreme_points) {
        if (point1 < point2) {
          extreme_p1.push_back(point1);
          extreme_p2.push_back(point2);
          std::vector<std::vector<double>> isoline_neighbor_length_tmp;
          std::vector<std::vector<Eigen::MatrixXd>> isoline_vertices_tmp;
          std::vector<std::vector<int>> isoline_face_ids_tmp;
          std::vector<double> isoline_length_tmp;
          std::vector<double> isoline_svs_tmp;
          std::vector<double> isoline_gs_tmp;
          std::vector<int> isoline_field_id_tmp;
          compute_segmentation_field(V, F, FN, E, L, point1, point2, num_fields, gradient_magnitude,
                                     isoline_neighbor_length_tmp,
                                     isoline_vertices_tmp, isoline_face_ids_tmp,
                                     isoline_length_tmp, isoline_gs_tmp, isoline_field_id_tmp, basic_tripletList,
                                     fields);
          std::vector<int> tmp_vec;
          field_to_isoline_ids.push_back(tmp_vec);
          for (int i = 0; i < isoline_vertices_tmp.size(); i++){
            isoline_neighbor_length.push_back(isoline_neighbor_length_tmp[i]);
            isoline_vertices.push_back(isoline_vertices_tmp[i]);
            isoline_face_ids.push_back(isoline_face_ids_tmp[i]);
            isoline_length.push_back(isoline_length_tmp[i]);
            isoline_gs.push_back(isoline_gs_tmp[i]);
            isoline_field_id.push_back(isoline_field_id_tmp[i]);
            field_to_isoline_ids.back().push_back(num_isos);
            num_isos++;
          }
          num_fields++;
        }
      }
    }
      end = std::chrono::steady_clock::now();
      std::cout << "FIELDS COMPUTED. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " sec\n";

    //TODO:
    //std::cout << gradient_magnitude[0].minCoeff() << std::endl;
    //std::cout << gradient_magnitude[0].maxCoeff() << std::endl;
    //std::cout << gradient_magnitude[1].minCoeff() << std::endl;
    //std::cout << gradient_magnitude[1].maxCoeff() << std::endl;

    //std::cout << (gradient_magnitude[0] - gradient_magnitude[1]).cwiseAbs().sum() << std::endl;
    //std::cout << (gradient_magnitude[0] - gradient_magnitude[2]).cwiseAbs().sum() << std::endl;
    //std::cout << (gradient_magnitude[1] - gradient_magnitude[2]).cwiseAbs().sum() << std::endl;

    std::vector<std::set<int>> isoline_ids_sharing_this_face_id(F.rows(), std::set<int>());

      //isoline_ids_sharing_this_face_id[0].insert(1);

      for (int i = 0; i < isoline_vertices.size(); i++){
          //std::set<int> face_set(isoline_face_ids[i].begin(), isoline_face_ids[i].end());
          //isoline_face_set.push_back(face_set);
          for (auto facetile: isoline_face_ids[i]) {
              isoline_ids_sharing_this_face_id[facetile].insert(i);
          }
      }

    ///Compute face sets for caching
    for (int i = 0; i < isoline_vertices.size(); i++){
      std::set<int> face_set(isoline_face_ids[i].begin(), isoline_face_ids[i].end());
      isoline_face_set.push_back(face_set);
    }

    num_isos = 0;
    /// Filter out bad isolines
    for (int i = 0; i < isoline_vertices.size(); i++){

      bool local_max = false;
      /// Check boundary conditions:
      double neigh_left_1_gs = 0;
      double neigh_left_2_gs = 0;
      double neigh_right_1_gs = 0;
      double neigh_right_2_gs = 0;
      bool neigh_left_1_same_id = false;
      bool neigh_left_2_same_id = false;
      bool neigh_right_1_same_id = false;
      bool neigh_right_2_same_id = false;
      if (i > 0){
        neigh_left_1_same_id = isoline_field_id[i] == isoline_field_id[i - 1];
        if (neigh_left_1_same_id)
          neigh_left_1_gs = isoline_gs[i - 1];
      }

      if (i > 0){
        neigh_left_2_same_id = isoline_field_id[i] == isoline_field_id[i - 2];
        if (neigh_left_2_same_id)
          neigh_left_2_gs = isoline_gs[i - 2];
      }

      if (i < isoline_vertices.size() - 1){
        neigh_right_1_same_id = isoline_field_id[i] == isoline_field_id[i + 1];
        if (neigh_right_1_same_id)
          neigh_right_1_gs = isoline_gs[i + 1];
      }

      if (i < isoline_vertices.size() - 2){
        neigh_right_2_same_id = isoline_field_id[i] == isoline_field_id[i + 2];
        if (neigh_right_2_same_id)
          neigh_right_2_gs = isoline_gs[i + 2];
      }

      if (neigh_left_2_gs <= isoline_gs[i] && neigh_left_1_gs <= isoline_gs[i] &&
          neigh_right_1_gs <= isoline_gs[i] && neigh_right_2_gs <= isoline_gs[i]){
        local_max = true;
      }

      /// Remove long isolines that share faces with short isolines
      bool is_longer_and_has_common_face = false;
      /*for (int j = 0; j < isoline_vertices.size(); j++){
        if (i != j){
          std::set<int> common;
          auto face_set1 = isoline_face_set[i];
          auto face_set2 = isoline_face_set[j];

          std::set_intersection (face_set1.begin(), face_set1.end(), face_set2.begin(), face_set2.end(),
                                 std::inserter(common,common.begin()));
          if (common.size() > 0){
            /// Check if length is similar to other isoline
            double l1 = isoline_length[i];
            double l2 = isoline_length[j];
            if (l1 > l2 * 1.5)
              is_longer_and_has_common_face = true;
          }
        }
      }*/

        ////// NEW ROUTINE
        for (auto face : isoline_face_set[i]){
            for (auto line: isoline_ids_sharing_this_face_id[face]){
                if (isoline_length[i] > isoline_length[line] * 1.5)
                    is_longer_and_has_common_face = true;
            }
        }

      /// Final Assignment
      if (!is_longer_and_has_common_face && local_max){
        candidate_vertices.push_back(isoline_vertices[i]);
        candidate_face_ids.push_back(isoline_face_ids[i]);
        candidate_gs.push_back(isoline_gs[i]);
        candidate_length.push_back(isoline_length[i]);
        candidate_neighbor_length.push_back(isoline_neighbor_length[i]);
        candidate_isoline_id.push_back(num_isos);
      }
      num_isos++;
    }

    isoline_local_gs = isoline_gs;

      end = std::chrono::steady_clock::now();
      std::cout << "ISOLINES COMPUTED. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " msec\n";

    /// Compute the Shape Variance Score
    compute_candidate_svs_new(candidate_length, candidate_svs, candidate_face_ids, F, V);
    //compute_candidate_svs_by_sdf(candidate_length, candidate_vertices, candidate_svs, candidate_face_ids, F, V);

    /// Compute the Gradient Magnitude Score
    compute_candidate_gs(gradient_magnitude, candidate_vertices, candidate_face_ids, candidate_length, candidate_gs, g_hat);

    /// Calculate Final Score
    candidate_score.resize(candidate_vertices.size());
    for (int i = 0; i < candidate_score.size(); i++){
      candidate_score[i] = candidate_gs[i] * candidate_svs[i];
    }

    create_edges_from_isolines(isoline_vertices, isoline_field_id, isoline_lines, num_fields);

    /// Find Valid cuts
    valid_cuts(V, N, F, dblA, IF, E, candidate_face_ids, candidate_score, applied_isolines);

    /// Create Edges from Isolines
    int num_edge = 0;
    for (int i = 0; i < applied_isolines.size(); i++){
      num_edge += candidate_vertices[applied_isolines[i]].size();
    }
    P1.resize(num_edge,3);
    P2.resize(num_edge,3);
    isoline_colors = Eigen::MatrixXd::Ones(num_edge, 3);
    int ind = 0;
    for (int i = 0; i < applied_isolines.size(); i++){
      Eigen::MatrixXd last_vertex;
      for (int j = 0; j < candidate_vertices[applied_isolines[i]].size(); j++){
        /// If first element: create edge to very last vertex
        if (j == 0)
          last_vertex = candidate_vertices[applied_isolines[i]].back();
        auto p1 = candidate_vertices[applied_isolines[i]][j];
        auto p2 = last_vertex;
        last_vertex = candidate_vertices[applied_isolines[i]][j];
        P1.row(ind) = p1;
        P2.row(ind) = p2;
        ind++;
      }
    }

    /// Create Edges Indices from Isolines
    num_edge = 0;
    int offset = 0;
    edge_indices.resize(isoline_vertices.size());
    for (int i = 0; i < isoline_vertices.size(); i++){
      offset += num_edge;
      num_edge = isoline_vertices[i].size();
      for (int j = offset; j < offset + num_edge; j++)
        edge_indices[i].push_back(j);
    }

    std::vector<std::vector<int>> segmentation_lines;
    for (int i = 0; i < applied_isolines.size(); i++){
      segmentation_lines.push_back(candidate_face_ids[applied_isolines[i]]);
    }


    color_mesh_by_isolines(E, F, segmentation_lines, vertex_labels);


    //use_cpc = true;
    /// MCPC Post-process:
    /// Run it for each part individually
      std::cout << "Max Label: " << vertex_labels.maxCoeff() << std::endl;
    if (use_cpc){
      for (int i = 0; i < vertex_labels.maxCoeff() + 1; i++){
        std::cout << "Post-Processing Part " << i << std::endl;
        std::set<int> vertex_list;
        for (int j = 0; j < V.rows(); j++){
          if ( vertex_labels(j,0) == (double) i )
            vertex_list.insert(j);
        }
        Eigen::MatrixXd SV;
        Eigen::MatrixXi SF;
        std::map<int, int> mfi2smfi;
        std::map<int, int> mvi2smvi;
        submesh(V, F, vertex_list, SV, SF, mvi2smvi, mfi2smfi);
        if (SF.rows() > 0){
          std::map<int, int> smfi2mfi;
          std::map<int, int> smvi2mvi;
          std::map<int, int> reversed;
          //Reverse the maps
          for (std::map<int, int>::iterator i = mfi2smfi.begin(); i != mfi2smfi.end(); ++i)
            smfi2mfi[i->second] = i->first;

          for (std::map<int, int>::iterator i = mvi2smvi.begin(); i != mvi2smvi.end(); ++i)
            smvi2mvi[i->second] = i->first;

          Eigen::MatrixXd labels_cpc;
          Eigen::MatrixXd labels_cpc_colors;
          cpc(SV, SF, labels_cpc, 0.0075f, 0.03f, 0.0f, 1.0f, 4.0f, false,
              false, 8.0, 0.1, 0, false, false, 0.16, 100, 1, true, true, true, 10000, 0,
              min_part_size * ( (float) V.rows() / (float) SV.rows()));
          Eigen::MatrixXi il, iu;
          Eigen::MatrixXd u;
          igl::unique(labels_cpc, u, il, iu);
          igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_JET, labels_cpc, true, labels_cpc_colors);
          std::cout << "Split Part into " << u.cols() * u.rows() << " Part(s) during CPC." << std::endl;
          //igl::writeOFF(mesh_file + "_part_" + std::to_string(i) + ".off",  SV, SF, labels_cpc_colors);

          /// Relabel Parts of CPC segmentation
          if (u.cols() * u.rows() > 1){
            for (int j = 0; j < u.cols() * u.rows(); j++){
              /// First: get next free label number
              double newLabel = vertex_labels.maxCoeff() + 1.0;
              for (int k = 0; k < SV.rows(); k++){
                if (labels_cpc(k,0) == u(j)){
                  vertex_labels(smvi2mvi[k],0) = newLabel;
                }
              }
            }
          }
        }
      }
    }

    //mergeSegments_old(V, F, vertex_labels, 0.02);
    //mergeSegments(V, F, vertex_labels, 0.02);

    /// Clean vertex_labels

    std::cout << "Max Label: " << vertex_labels.maxCoeff() << std::endl;
    Eigen::MatrixXi IA, IC;
    Eigen::MatrixXd CC;
    igl::unique(vertex_labels, CC, IA, IC);
    vertex_labels = IC.cast <double> ();

    mesh_label_colored.resize(V.rows(), 3);

    Eigen::VectorXd col_palette = Eigen::VectorXd::LinSpaced(vertex_labels.maxCoeff() + 1, 0.0, 360.0 - (360.0 / (vertex_labels.maxCoeff() + 1)));

    if (vertex_labels.maxCoeff() > 0){
      for (int i = 0; i <= vertex_labels.maxCoeff(); i++){
        /// Create new random color
        //double h = dis(gen) * 360.0;
        double h = col_palette(i);
        double s = 1.0;
        double v = 1.0;
        hsv c1;
        c1.h = h;
        c1.s = s;
        c1.v = v;
        rgb c2 = hsv2rgb(c1);
        double r = c2.r;
        double g = c2.g;
        double b = c2.b;
        for (int j = 0; j < V.rows(); j++){
          if (vertex_labels(j,0) == i){
            mesh_label_colored(j,0) = r;
            mesh_label_colored(j,1) = g;
            mesh_label_colored(j,2) = b;
          }
        }
      }
    }

    /// Write .seg file
    Eigen::MatrixXi face_labels = Eigen::MatrixXi::Zero(F.rows(), 1);
    for (int i = 0; i < F.rows(); i++){
      /// Check the three labels of the vertices
      int label1 = vertex_labels(F(i,0));
      int label2 = vertex_labels(F(i,1));
      int label3 = vertex_labels(F(i,2));
      /// If all three labels are eqaul: Assign face to this label
      if ( (label1 == label2) && (label2 && label3) ){
        face_labels(i,0) = label1;
      }
        /// If two of the triangle's vertices have the same label
      else if ( (label1 == label2 || label1 == label3) ){
        face_labels(i,0) = label1;
      }
      else if ( (label1 == label2 || label2 == label3 ) ){
        face_labels(i,0) = label2;
      }
      else if ( (label1 == label3 || label2 == label3 ) ){
        face_labels(i,0) = label3;
      }
        /// The three labels are all different, just assign the face to the one with the smallest label_id
      else{
        if (label1 <= label2 && label1 <= label3 ){
          face_labels(i,0) = label1;
        }else if (label2 <= label1 && label2 <= label3 ){
          face_labels(i,0) = label2;
        }else if (label3 <= label1 && label3 <= label2 ){
          face_labels(i,0) = label3;
        }
      }
    }

    Eigen::MatrixXi IAi, ICi, CCi;
    Eigen::MatrixXi face_labels_tmp = face_labels;
    igl::unique(face_labels_tmp, CCi, IAi, ICi);
    face_labels = ICi;

    igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_JET, vertex_labels, true, C);

    //std::size_t found = mesh_file.find_last_of("/");
    //std::string filename = mesh_file.substr(found + 1, mesh_file.size()-4);
    //std::string out_path = out + filename;
    //std::string seg_out = out_path + ".seg";
    //std::string mesh_out = out_path + "_labeled.off";
    //std::string pc_out = out_path + ".pcd";
    //std::cout << "out=" << out << std::endl;
    //std::cout << "filename=" << filename << std::endl;
    //std::cout << "mesh_file=" << mesh_file << std::endl;
    //std::cout << "out_path=" << out_path << std::endl;
    //std::cout << "seg_out=" << out << std::endl;
    //std::cout << "mesh_out=" << mesh_out << std::endl;
    //std::cout << "pc_out=" << pc_out << std::endl;
    /*pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
    for (int i = 0; i < vertex_labels.rows(); i++){
      pcl::PointXYZL p1;
      p1.x = V(i,0);
      p1.y = V(i,1);
      p1.z = V(i,2);
      p1.label = vertex_labels(i,0);
      cloud->push_back(p1);
    }*/
    //pcl::io::savePCDFile(pc_out, *cloud, 0);
    //igl::writeOFF(mesh_out, V, F, mesh_label_colored);
    /*
    for (int i = 0; i < face_labels.maxCoeff() + 1; i++){
      int num_faces = (face_labels.array() == i).count();
      int curr_ind = 0;
      Eigen::MatrixXi F_tmp = Eigen::MatrixXi::Zero(num_faces, 3);
      for (int j = 0; j < face_labels.rows(); j++){
        if (face_labels(j,0) == i){
          F_tmp.row(curr_ind) = F.row(j);
          curr_ind++;
        }
      }
      Eigen::MatrixXi V_unique;
      Eigen::MatrixXi IA2, IC2;
      igl::unique(F_tmp, V_unique, IA2, IC2);
      int num_vert = V_unique.rows() * V_unique.cols();
      Eigen::MatrixXd V_tmp = Eigen::MatrixXd::Zero(num_vert,3);
      for (int j = 0; j < num_vert; j++){
        V_tmp.row(j) = V.row(V_unique(j,0));
      }
      Eigen::MatrixXi F_tmp2 = Eigen::MatrixXi::Zero(num_faces, 3);
      for (int j = 0; j < F_tmp2.rows(); j++){
        F_tmp2(j,0) = IC2(j,0);
        F_tmp2(j,1) = IC2(j + num_faces,0);
        F_tmp2(j,2) = IC2(j + 2 * num_faces,0);
      }
      std::string out_part = seg_out + "_part_" + std::to_string(i) + ".off";
    }*/

    std::ofstream myfile;
    myfile.open (out);
    myfile << face_labels;
    myfile.close();

    //end = std::chrono::steady_clock::now();
    //std::cout << "Elapsed time in seconds : "
    //            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //            << " sec\n";

    /// Export individual Parts
    /*
    for (int i = 0; i < vertex_labels.maxCoeff() + 1; i++){
      Eigen::MatrixXi F_tmp = Eigen::MatrixXi::Zero(0,0);
      Eigen::MatrixXd V_tmp = Eigen::MatrixXd::Zero((vertex_labels.array() == i).count(), 3);
      int curr_ind = 0;
      for (int j = 0; j < V.rows(); j++){
        if (vertex_labels(j,0) == i){
          V_tmp.row(curr_ind) = V.row(j);
          curr_ind++;
        }
      }
      std::string mesh_out2 = out_path + "_part_" + std::to_string(i) + "_labeled.off";
      //igl::writeOFF(mesh_out2, V_tmp, F_tmp);
    }*/

    draw_isoline = true;

    /// Get id of mesh seg file
    int id_end = mesh_file.find_last_of(".");
    std::string seg_id = mesh_file.substr(0, id_end);
    int id_begin = mesh_file.find_last_of("/");
    seg_id = seg_id.substr(id_begin + 1);
    //std::cout << seg_id << std::endl;
    //res = evaluate_segmentation(seg_id, out_path + ".seg", "./");

  };

    //if (vm.count("nogui")){

    //f();

    //}else{
    /// Visualization
    std::cout << "test" << std::endl;

    // Init the viewer
    igl::opengl::glfw::Viewer viewer;

    if (vm.count("nogui")) {
        GLFWwindow *window;
        igl::opengl::glfw::background_window(window);
        glfwSetWindowSize(window, 200, 200);
        viewer.window = window;
    }

    viewer.core().is_animating = true;

    // Attach a custom menu
    CustomMenu menu;
    viewer.plugins.push_back(&menu);

    menu.visualization_choices = {"Standard", "Field", "Field Gradient", "Concavity"};
    menu.visualization_choice = 0;
    menu.label_choices = {"None", "Local Gradient Score", "SVS", "Gradient Score", "Score", "ID"};
    menu.label_choice = 0;
    menu.sigma = sigma;
    menu.beta = beta;
    menu.epsilon = eps;
    menu.run_segmentation = false;
    menu.use_cpc = use_cpc;
    menu.draw_iso = draw_isoline;
    menu.redraw = true;

    viewer.data().set_mesh(V, F);

    /// Pre Draw Callback
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &)
    {
      if(menu.redraw) {
        if (menu.run_segmentation){
          std::cout << "Running Segmentation...\n";
          menu.run_segmentation = false;
          beta = menu.beta;
          eps = menu.epsilon;
          sigma = menu.sigma;
          use_cpc = menu.use_cpc;
          draw_isoline = menu.draw_iso;
          f();

          std::vector<std::string> items;
          for (int i = 0; i < num_fields; i++){
            items.push_back({ "Field " + std::to_string(i + 1)});
          }
          items.push_back("Normalized");
          gradient_magnitude.push_back(g_hat);
          menu.field_choices = items;
          menu.performance = res;

          menu.segmentation_finished = true;

            if (vm.count("nogui")) {
                viewer.launch_shut();
            }
        }
        if (menu.segmentation_finished){
          switch (menu.visualization_choice) {
            case 0:
              ///Standard Mode
              viewer.data().clear();
              viewer.data().set_mesh(V, F);
              if (vertex_labels.rows() == V.rows()){
                viewer.data().set_colors(mesh_label_colored);
              }
              redraw = false;
              break;

            case 1:
              ///Field Mode
              viewer.data().clear();
              viewer.data().set_mesh(V, F);
              if (menu.field_choice < gradient_magnitude.size() - 1) {
                igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_JET, fields[menu.field_choice], true, C);
                viewer.data().set_colors(C);
              }
              redraw = false;
              break;

            case 2:
              ///Field Gradient Mode
              viewer.data().clear();
              viewer.data().set_mesh(V, F);
              if (menu.field_choice < gradient_magnitude.size()) {
                igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_PARULA,
                              gradient_magnitude[menu.field_choice],
                              true, C);
                viewer.data().set_colors(C);
              }
              redraw = false;
              break;

            case 3:
              ///Concavity Mode
              viewer.data().clear();
              viewer.data().set_mesh(V, F);
              igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_PARULA, vertex_is_concave, true, C);
              viewer.data().set_colors(C);
              redraw = false;
              break;

          }

          ///Draw extreme Points
          if (menu.field_choice < menu.field_choices.size() - 1 && menu.field_choices.size() > 0) {
            viewer.data().add_points(V.row(extreme_p1[menu.field_choice]), Eigen::RowVector3d(1.0, 0.0, 0.0));
            viewer.data().add_points(V.row(extreme_p2[menu.field_choice]), Eigen::RowVector3d(0.0, 0.0, 1.0));
          } else {
            for (int i = 0; i < extreme_points.size(); i++) {
              viewer.data().add_points(V.row(extreme_points[i]), Eigen::RowVector3d(1.0, 1.0, 1.0));
            }
          }

          if (draw_isoline) {
            /// Final Cuts, when Normalized Field is selected
            if (isoline_lines.size() == menu.field_choice) {
              viewer.data().add_edges(P1, P2, isoline_colors);

              if (menu.label_choice == 5) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0), std::to_string(i));
                }
              } else if (menu.label_choice == 4) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0),
                                          std::to_string(candidate_score[applied_isolines[i]]));
                }
              } else if (menu.label_choice == 2) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0),
                                          std::to_string(candidate_svs[applied_isolines[i]]));
                }
              } else if (menu.label_choice == 1) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0),
                                          std::to_string(
                                            isoline_local_gs[candidate_isoline_id[applied_isolines[i]]]));
                }
              } else if (menu.label_choice == 3) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0),
                                          std::to_string(candidate_gs[applied_isolines[i]]));
                }
              }
            }

            /// All Isolines, when some Segmentation Field is selected
            if (isoline_lines.size() > menu.field_choice) {

              for (int i = 0; i < field_to_isoline_ids[menu.field_choice].size(); i++) {
                int iso_id = field_to_isoline_ids[menu.field_choice][i];
                /// Get Candidate ID if isoline is indeed a candidate
                auto it = std::find(candidate_isoline_id.begin(), candidate_isoline_id.end(), iso_id);
                int index = -1;
                if (it == candidate_isoline_id.end()) {
                  /// Isoline is not a candidate
                } else {
                  index = std::distance(candidate_isoline_id.begin(), it);
                }

                /// Add each isoline individually
                Eigen::MatrixXd tmp_p1 = Eigen::MatrixXd::Zero(edge_indices[iso_id].size(), 3);
                Eigen::MatrixXd tmp_p2 = Eigen::MatrixXd::Zero(edge_indices[iso_id].size(), 3);
                Eigen::MatrixXd last_vertex;
                for (int j = 0; j < edge_indices[iso_id].size(); j++) {
                  if (j == 0)
                    last_vertex = isoline_vertices[iso_id].back();
                  auto p1 = isoline_vertices[iso_id][j];
                  auto p2 = last_vertex;
                  last_vertex = isoline_vertices[iso_id][j];
                  tmp_p1.row(j) = p1;
                  tmp_p2.row(j) = p2;
                }
                Eigen::MatrixXd tmp_c = Eigen::MatrixXd::Ones(edge_indices[iso_id].size(), 3);
                if (std::find(candidate_isoline_id.begin(), candidate_isoline_id.end(), iso_id) !=
                    candidate_isoline_id.end() && show_candidates) {
                  for (int j = 0; j < tmp_c.rows(); j++) {
                    tmp_c.row(j) << 1.0, 0.0, 1.0;
                  }
                }
                //viewer.data().add_edges(tmp_p1, tmp_p2, tmp_c);

                /// Add Labels
                if (menu.label_choice == 1) {
                  viewer.data().add_label(isoline_vertices[iso_id][0].row(0),
                                          std::to_string(isoline_local_gs[iso_id]));
                } else if (menu.label_choice == 5) {
                  if (index != -1)
                    viewer.data().add_label(isoline_vertices[iso_id][0].row(0), std::to_string(index));
                } else if (menu.label_choice == 2) {
                  if (index != -1)
                    viewer.data().add_label(isoline_vertices[iso_id][0].row(0), std::to_string(candidate_svs[index]));
                } else if (menu.label_choice == 4) {
                  if (index != -1)
                    viewer.data().add_label(isoline_vertices[iso_id][0].row(0),
                                            std::to_string(candidate_score[index]));
                } else if (menu.label_choice == 3) {
                  if (index != -1)
                    viewer.data().add_label(isoline_vertices[iso_id][0].row(0), std::to_string(candidate_gs[index]));
                }
              }
            }
          }
        }

      }
      return false;
    };

    /// start viewer
    if (vm.count("nogui")) {
        menu.run_segmentation = true;
    }
    viewer.launch();

  //std::cout << ">> Algorithm ended.\n";
  return 0;
}