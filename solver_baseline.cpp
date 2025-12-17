#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "baseline.hpp"
#include "geom.hpp"

namespace {

constexpr int kOutputDecimals = 9;

double quantize_value(double x) {
    const std::string s = fmt_submission_value(x, kOutputDecimals);
    return std::stod(s.substr(1));
}

TreePose quantize_pose(const TreePose& pose) {
    return TreePose{quantize_value(pose.x),
                    quantize_value(pose.y),
                    quantize_value(pose.deg)};
}

std::vector<TreePose> quantize_poses(const std::vector<TreePose>& poses) {
    std::vector<TreePose> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back(quantize_pose(p));
    }
    return out;
}

double score_instance(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    auto polys = transformed_polygons(base_poly, poses);
    double s = bounding_square_side(polys);
    return (s * s) / static_cast<double>(poses.size());
}

}  // namespace

int main() {
    try {
        Polygon base_poly = get_tree_polygon();
        BaselineConfig cfg;

        const std::string output_path = "submission_baseline_cpp.csv";
        std::ofstream out(output_path);
        if (!out) {
            std::cerr << "Erro ao abrir arquivo de saída: " << output_path
                      << "\n";
            return 1;
        }

        out << "id,x,y,deg\n";

        double total_score = 0.0;

        for (int n = 1; n <= 200; ++n) {
            auto poses = generate_grid_poses_for_n(n, base_poly, cfg);
            auto poses_q = quantize_poses(poses);
            total_score += score_instance(base_poly, poses_q);
            for (int i = 0; i < n; ++i) {
                const auto& pose = poses_q[i];
                out << std::setw(3) << std::setfill('0') << n << "_" << i
                    << "," << fmt_submission_value(pose.x) << ","
                    << fmt_submission_value(pose.y) << ","
                    << fmt_submission_value(pose.deg) << "\n";
            }
        }

        std::cout << "Submission gerada em " << output_path << "\n";
        std::cout << "Score (local): " << std::fixed << std::setprecision(9)
                  << total_score << "\n";
    } catch (const std::exception &ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
