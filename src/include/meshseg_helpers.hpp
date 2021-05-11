#include "tinyxml2.h"
#include <cstdlib>
#include <pstream.h>
#include <iostream>
#include <boost/filesystem.hpp>

using namespace tinyxml2;

/**
 *
 * @param mesh_id
 * @param seg_path
 * @param out_path
 * @return
 */
std::vector<double> evaluate_segmentation(std::string mesh_id, std::string seg_path, std::string out_path) {
    XMLDocument doc;
    doc.LoadFile("/home/florian/FunctionalObjectUnderstanding/settings.xml");
    std::string content = doc.FirstChildElement("settings")->FirstChildElement("meshseg_path")->GetText();
    std::string temp_path = doc.FirstChildElement("settings")->FirstChildElement("temp_path")->GetText();

    double CD = 0.0;
    double CE = 0.0;
    double CE_0 = 0.0;
    double CE_1 = 0.0;
    double CE_2 = 0.0;
    double CE_3 = 0.0;
    double HD = 0.0;
    double HD_0 = 0.0;
    double HD_1 = 0.0;
    double HD_2 = 0.0;
    double RI = 0.0;
    std::vector<double> HD_vec;
    std::vector<double> CE_vec;
    int num_seg_files = 0;
    double x;

    std::string eval_exe_path = content + "exe/segEval";
    std::string mesh_path = content + "data/off/" + mesh_id;
    std::string seg_benchmark_path = content + "data/seg/Benchmark/" + mesh_id + "/";

    std::vector<std::string> seg_benchmark;

    if (boost::filesystem::is_directory(seg_benchmark_path))
        for (auto &p : boost::filesystem::directory_iterator(seg_benchmark_path))
            seg_benchmark.push_back(p.path().string());

    num_seg_files = seg_benchmark.size();
    /// Loop over this list and keep track of the scores
    for (auto seg_file : seg_benchmark) {
        /// For each seg file from the benchmark: strip away everything except for the core name
        std::size_t name_begin = seg_file.find_last_of("/");
        std::string name_seg = seg_file.substr(name_begin + 1);
        std::size_t name_end = name_seg.find_last_of(".");
        name_seg = name_seg.substr(0, name_end);
        //std::cout << name_seg << std::endl;

        std::string cmd = eval_exe_path + " " + mesh_path + ".off " + seg_path + " " + seg_file + " " + temp_path;
        redi::ipstream proc(cmd, redi::pstreams::pstdout | redi::pstreams::pstderr);
        std::string line;
        //std::cout << cmd << '\n';
        while (std::getline(proc.out(), line)) {};

        std::ifstream inFile;
        inFile.open(temp_path + name_seg + ".CD");
        while (inFile >> x) {
            CD = CD + x;
        }
        inFile.close();

        inFile.open(temp_path + name_seg + ".RI");
        while (inFile >> x) {
            RI = RI + x;
        }
        inFile.close();

        inFile.open(temp_path + name_seg + ".CE");
        while (inFile >> x) {
            CE_vec.push_back(x);
        }
        inFile.close();

        inFile.open(temp_path + name_seg + ".HD");
        while (inFile >> x) {
            HD_vec.push_back(x);
        }
        inFile.close();
    }

    for (int i = 0; i < CE_vec.size(); i++) {
        if (i % 4 == 0)
            CE_0 += CE_vec[i];
        if (i % 4 == 1)
            CE_1 += CE_vec[i];
        if (i % 4 == 2)
            CE_2 += CE_vec[i];
        if (i % 4 == 3)
            CE_3 += CE_vec[i];
    }

    for (int i = 0; i < HD_vec.size(); i++) {
        if (i % 3 == 0)
            HD_0 += CE_vec[i];
        if (i % 3 == 1)
            HD_1 += CE_vec[i];
        if (i % 3 == 2)
            HD_2 += CE_vec[i];
    }

    HD_0 = HD_0 / (double) num_seg_files;
    HD_1 = HD_1 / (double) num_seg_files;
    HD_2 = HD_2 / (double) num_seg_files;
    CE_0 = CE_0 / (double) num_seg_files;
    CE_1 = CE_1 / (double) num_seg_files;
    CE_2 = CE_2 / (double) num_seg_files;
    CE_3 = CE_3 / (double) num_seg_files;
    RI = RI / (double) num_seg_files;
    CD = CD / (double) num_seg_files;

    /// At the end: return the performance
    std::cout << "Segmentation Performance Report:\n";
    std::cout << "HD: " << HD_0 << std::endl;
    std::cout << "CD: " << CD << std::endl;
    std::cout << "CE: " << CE_0 << std::endl;
    std::cout << "RI: " << RI << std::endl;

    std::vector<double> res = {HD_0, HD_1, HD_2, CE_0, CE_1, CE_2, CE_3, RI, CD};
    return res;
}