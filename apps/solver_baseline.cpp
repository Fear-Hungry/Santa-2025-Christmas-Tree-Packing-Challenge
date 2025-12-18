#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "baseline.hpp"
#include "geom.hpp"
#include "submission_io.hpp"

namespace {

constexpr int kOutputDecimals = 9;

double score_instance(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    auto polys = transformed_polygons(base_poly, poses);
    double s = bounding_square_side(polys);
    return (s * s) / static_cast<double>(poses.size());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Polygon base_poly = get_tree_polygon();
        BaselineConfig cfg;

        std::string output_path = "submission_baseline_cpp.csv";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--output") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--output precisa de PATH.");
                }
                output_path = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Uso: " << argv[0]
                          << " [--output PATH]\n";
                return 0;
            } else {
                throw std::runtime_error("Argumento desconhecido: " + arg);
            }
        }

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
            auto poses_q = quantize_poses(poses, kOutputDecimals);
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
