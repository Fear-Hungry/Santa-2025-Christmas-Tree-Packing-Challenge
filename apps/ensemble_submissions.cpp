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

long double term_for_side(double side, int n) {
    const long double s = static_cast<long double>(side);
    return (s * s) / static_cast<long double>(n);
}

struct DebugEntry {
    int k = -1;
    long double pre_term = 0.0L;
    long double post_term = 0.0L;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Uso: " << argv[0]
                << " output.csv input1.csv [input2.csv ...] [--no-final-rigid]"
                   " [--debug-n N] [--debug-top K]\n";
            return 2;
        }

        const std::string output_path = argv[1];
        bool final_rigid = true;
        int debug_n = -1;
        int debug_top = 5;
        std::vector<std::string> inputs;
        auto need = [&](const std::string& name, int& i) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Faltou valor para " + name);
            }
            return argv[++i];
        };
        auto parse_int = [](const std::string& s, const std::string& name) -> int {
            size_t pos = 0;
            int v = std::stoi(s, &pos);
            if (pos != s.size()) {
                throw std::runtime_error("Inteiro inválido para " + name + ": " + s);
            }
            return v;
        };
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--no-final-rigid" || arg == "--no-sa-rigid") {
                final_rigid = false;
                continue;
            }
            if (arg == "--debug-n") {
                debug_n = parse_int(need(arg, i), arg);
                continue;
            }
            if (arg == "--debug-top") {
                debug_top = parse_int(need(arg, i), arg);
                continue;
            }
            inputs.push_back(std::move(arg));
        }
        if (inputs.empty()) {
            throw std::runtime_error("Precisa de ao menos 1 input.");
        }
        if (debug_n != -1 && (debug_n < 1 || debug_n > 200)) {
            throw std::runtime_error("--debug-n precisa estar em [1,200] ou -1.");
        }
        if (debug_top <= 0) {
            throw std::runtime_error("--debug-top precisa ser > 0.");
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
        const long double kEvalEps = 1e-18L;
        const double kSideEps = 1e-15;

        for (int n = 1; n <= 200; ++n) {
            int best_k = -1;
            long double best_eval_term = std::numeric_limits<long double>::infinity();
            std::vector<TreePose> best_poses;
            const bool debug_this_n = (debug_n == n);
            std::vector<DebugEntry> debug_entries;
            if (debug_this_n) {
                debug_entries.reserve(subs.size());
            }

            for (int k = 0; k < static_cast<int>(subs.size()); ++k) {
                const auto& poses = subs[static_cast<size_t>(k)].by_n[static_cast<size_t>(n)];
                if (any_overlap(base_poly, poses, radius)) {
                    continue;
                }
                double pre_side = side_for_solution(base_poly, poses);
                long double pre_term = term_for_side(pre_side, n);
                double post_side = pre_side;
                long double post_term = pre_term;

                if (final_rigid) {
                    std::vector<TreePose> rigid_sol = poses;
                    optimize_rigid_rotation(base_poly, rigid_sol);
                    auto rigid_q = quantize_poses(rigid_sol);
                    if (!any_overlap(base_poly, rigid_q, radius)) {
                        double rigid_side = side_for_solution(base_poly, rigid_q);
                        if (rigid_side + kSideEps < post_side) {
                            post_side = rigid_side;
                            post_term = term_for_side(post_side, n);
                        }
                    }
                }

                if (debug_this_n) {
                    debug_entries.push_back(DebugEntry{k, pre_term, post_term});
                }

                if (post_term + kEvalEps < best_eval_term ||
                    (std::fabsl(post_term - best_eval_term) <= kEvalEps && k < best_k)) {
                    best_eval_term = post_term;
                    best_k = k;
                    best_poses = std::move(poses);
                }
            }

            if (best_k < 0) {
                throw std::runtime_error("Nenhum input válido para n=" + std::to_string(n));
            }

            if (debug_this_n) {
                auto print_top = [&](const char* label,
                                     const std::vector<DebugEntry>& entries,
                                     bool use_post) {
                    const int k = std::min(debug_top, static_cast<int>(entries.size()));
                    std::cerr << "[debug] n=" << n << " top-" << k << " " << label << "\n";
                    std::cerr << "rank,k,term,input\n";
                    for (int i = 0; i < k; ++i) {
                        const auto& e = entries[static_cast<size_t>(i)];
                        long double t = use_post ? e.post_term : e.pre_term;
                        std::cerr << (i + 1) << ","
                                  << e.k << ","
                                  << std::fixed << std::setprecision(12)
                                  << static_cast<double>(t) << ","
                                  << inputs[static_cast<size_t>(e.k)] << "\n";
                    }
                };
                auto by_pre = debug_entries;
                auto by_post = debug_entries;
                std::sort(by_pre.begin(),
                          by_pre.end(),
                          [](const DebugEntry& a, const DebugEntry& b) {
                              if (a.pre_term != b.pre_term) {
                                  return a.pre_term < b.pre_term;
                              }
                              return a.k < b.k;
                          });
                std::sort(by_post.begin(),
                          by_post.end(),
                          [](const DebugEntry& a, const DebugEntry& b) {
                              if (a.post_term != b.post_term) {
                                  return a.post_term < b.post_term;
                              }
                              return a.k < b.k;
                          });
                print_top("pre-rigid (term)", by_pre, false);
                print_top("post-rigid (term)", by_post, true);
                auto chosen_it = std::find_if(debug_entries.begin(),
                                              debug_entries.end(),
                                              [&](const DebugEntry& e) { return e.k == best_k; });
                if (chosen_it != debug_entries.end()) {
                    std::cerr << "[debug] chosen k=" << best_k
                              << " pre_term=" << std::fixed << std::setprecision(12)
                              << static_cast<double>(chosen_it->pre_term)
                              << " post_term=" << static_cast<double>(chosen_it->post_term)
                              << " input=" << inputs[static_cast<size_t>(best_k)]
                              << "\n";
                }
            }

            std::vector<TreePose> out_poses = best_poses;
            double out_side = side_for_solution(base_poly, out_poses);
            if (final_rigid) {
                std::vector<TreePose> rigid_sol = out_poses;
                optimize_rigid_rotation(base_poly, rigid_sol);
                auto rigid_q = quantize_poses(rigid_sol);
                if (!any_overlap(base_poly, rigid_q, radius)) {
                    double rigid_side = side_for_solution(base_poly, rigid_q);
                    if (rigid_side + kSideEps < out_side) {
                        out_side = rigid_side;
                        out_poses = std::move(rigid_q);
                    }
                }
            }

            total_score += (out_side * out_side) / static_cast<double>(n);

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
