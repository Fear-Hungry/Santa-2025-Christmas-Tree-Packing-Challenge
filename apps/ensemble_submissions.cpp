#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"
#include "submission_io.hpp"

namespace {

double side_for_solution(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    auto polys = transformed_polygons(base_poly, poses);
    return bounding_square_side(polys);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Uso: " << argv[0]
                << " output.csv input1.csv [input2.csv ...] [--no-final-rigid]\n";
            return 2;
        }

        const std::string output_path = argv[1];
        bool final_rigid = true;
        std::vector<std::string> inputs;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--no-final-rigid" || arg == "--no-sa-rigid") {
                final_rigid = false;
                continue;
            }
            inputs.push_back(std::move(arg));
        }
        if (inputs.empty()) {
            throw std::runtime_error("Precisa de ao menos 1 input.");
        }

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);

        std::vector<SubmissionPoses> subs;
        subs.reserve(inputs.size());
        for (const auto& p : inputs) {
            subs.push_back(load_submission_poses(p, 200));
        }

        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("Erro ao abrir arquivo de saída: " + output_path);
        }
        out << "id,x,y,deg\n";

        double total_score = 0.0;

        for (int n = 1; n <= 200; ++n) {
            int best_k = -1;
            double best_side = std::numeric_limits<double>::infinity();
            std::vector<TreePose> best_poses;

            for (int k = 0; k < static_cast<int>(subs.size()); ++k) {
                const auto& poses = subs[static_cast<size_t>(k)].by_n[static_cast<size_t>(n)];
                if (any_overlap(base_poly, poses, radius)) {
                    continue;
                }
                double side = side_for_solution(base_poly, poses);
                if (side + 1e-15 < best_side) {
                    best_side = side;
                    best_k = k;
                    best_poses = std::move(poses);
                }
            }

            if (best_k < 0) {
                throw std::runtime_error("Nenhum input válido para n=" + std::to_string(n));
            }

            std::vector<TreePose> out_poses = best_poses;
            if (final_rigid) {
                std::vector<TreePose> rigid_sol = out_poses;
                optimize_rigid_rotation(base_poly, rigid_sol);
                auto rigid_q = quantize_poses(rigid_sol);
                if (!any_overlap(base_poly, rigid_q, radius)) {
                    double rigid_side = side_for_solution(base_poly, rigid_q);
                    if (rigid_side + 1e-15 < best_side) {
                        best_side = rigid_side;
                        out_poses = std::move(rigid_q);
                    }
                }
            }

            total_score += (best_side * best_side) / static_cast<double>(n);

            for (int i = 0; i < n; ++i) {
                const auto& p = out_poses[static_cast<size_t>(i)];
                out << std::setw(3) << std::setfill('0') << n << "_" << i << ","
                    << fmt_submission_value(p.x) << ","
                    << fmt_submission_value(p.y) << ","
                    << fmt_submission_value(p.deg) << "\n";
            }
        }

        std::cout << "Submission ensembling gerada em " << output_path << "\n";
        std::cout << "Inputs: " << inputs.size() << "\n";
        std::cout << "Score (local): " << std::fixed << std::setprecision(9)
                  << total_score << "\n";
        std::cout << "Final rigid: " << (final_rigid ? "on" : "off") << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
