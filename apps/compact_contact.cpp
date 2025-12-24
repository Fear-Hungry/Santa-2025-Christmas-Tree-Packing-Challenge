#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "solvers/compaction_contact.hpp"
#include "geometry/geom.hpp"
#include "utils/submission_io.hpp"

namespace {

struct Options {
    std::string base_path;
    std::string output_path = "runs/tmp/compact_contact.csv";
    int n_min = 1;
    int n_max = 200;
    int target_top = 0;
    std::vector<int> target_ns;
    int target_range_min = 1;
    int target_range_max = 200;
    bool target_range_set = false;
    uint64_t seed = 1;
    int passes = 30;
    int attempts_per_pass = 48;
    int patience = 4;
    int boundary_topk = 12;
    int push_bisect_iters = 10;
    double push_max_step_frac = 1.0;
    double plateau_eps = 0.0;
    bool alt_axis = true;
    double diag_frac = 0.0;
    double diag_rand = 0.0;
    double center_bias = 0.0;
    double interior_prob = 0.0;
    double shake_pos = 0.0;
    double shake_rot_deg = 0.0;
    double shake_prob = 1.0;
    bool final_rigid = true;
    int quantize_decimals = 9;
};

double score_term_for_n(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    if (poses.empty()) {
        return 0.0;
    }
    std::vector<Polygon> polys = transformed_polygons(base_poly, poses);
    const double side = bounding_square_side(polys);
    return (side * side) / static_cast<double>(poses.size());
}

Options parse_args(int argc, char** argv) {
    Options opt;
    auto parse_int = [](const std::string& s) -> int {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Inteiro invalido: " + s);
        }
        return v;
    };
    auto parse_u64 = [](const std::string& s) -> uint64_t {
        size_t pos = 0;
        uint64_t v = std::stoull(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("uint64 invalido: " + s);
        }
        return v;
    };
    auto parse_double = [](const std::string& s) -> double {
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Double invalido: " + s);
        }
        return v;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Faltou valor para " + name);
            }
            return argv[++i];
        };

        if (arg == "--base") {
            opt.base_path = need(arg);
        } else if (arg == "--out" || arg == "--output") {
            opt.output_path = need(arg);
        } else if (arg == "--target-top") {
            opt.target_top = parse_int(need(arg));
        } else if (arg == "--target-n") {
            std::string s = need(arg);
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) {
                    continue;
                }
                opt.target_ns.push_back(parse_int(item));
            }
        } else if (arg == "--target-range") {
            std::string s = need(arg);
            std::stringstream ss(s);
            std::string a, b;
            if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',')) {
                throw std::runtime_error("--target-range precisa ser 'a,b'.");
            }
            opt.target_range_min = parse_int(a);
            opt.target_range_max = parse_int(b);
            opt.target_range_set = true;
        } else if (arg == "--n-min") {
            opt.n_min = parse_int(need(arg));
        } else if (arg == "--n-max") {
            opt.n_max = parse_int(need(arg));
        } else if (arg == "--seed") {
            opt.seed = parse_u64(need(arg));
        } else if (arg == "--passes") {
            opt.passes = parse_int(need(arg));
        } else if (arg == "--attempts-per-pass") {
            opt.attempts_per_pass = parse_int(need(arg));
        } else if (arg == "--patience") {
            opt.patience = parse_int(need(arg));
        } else if (arg == "--boundary-topk") {
            opt.boundary_topk = parse_int(need(arg));
        } else if (arg == "--push-bisect-iters") {
            opt.push_bisect_iters = parse_int(need(arg));
        } else if (arg == "--push-max-step-frac") {
            opt.push_max_step_frac = parse_double(need(arg));
        } else if (arg == "--plateau-eps") {
            opt.plateau_eps = parse_double(need(arg));
        } else if (arg == "--diag-frac") {
            opt.diag_frac = parse_double(need(arg));
        } else if (arg == "--diag-rand") {
            opt.diag_rand = parse_double(need(arg));
        } else if (arg == "--center-bias") {
            opt.center_bias = parse_double(need(arg));
        } else if (arg == "--interior-prob") {
            opt.interior_prob = parse_double(need(arg));
        } else if (arg == "--shake-pos") {
            opt.shake_pos = parse_double(need(arg));
        } else if (arg == "--shake-rot-deg") {
            opt.shake_rot_deg = parse_double(need(arg));
        } else if (arg == "--shake-prob") {
            opt.shake_prob = parse_double(need(arg));
        } else if (arg == "--no-alt-axis") {
            opt.alt_axis = false;
        } else if (arg == "--no-final-rigid") {
            opt.final_rigid = false;
        } else if (arg == "--quantize-decimals") {
            opt.quantize_decimals = parse_int(need(arg));
        } else {
            throw std::runtime_error("Flag desconhecida: " + arg);
        }
    }

    if (opt.base_path.empty()) {
        throw std::runtime_error("Use --base <csv>.");
    }
    if (opt.n_min < 1 || opt.n_min > opt.n_max || opt.n_max > 200) {
        throw std::runtime_error("--n-min/--n-max invalidos.");
    }
    if (opt.target_top < 0) {
        throw std::runtime_error("--target-top precisa ser >= 0.");
    }
    for (int n : opt.target_ns) {
        if (n < 1 || n > 200) {
            throw std::runtime_error("--target-n fora de [1,200].");
        }
    }
    if (opt.target_range_set) {
        if (opt.target_range_min < 1 || opt.target_range_max > 200 ||
            opt.target_range_min > opt.target_range_max) {
            throw std::runtime_error("--target-range invalido.");
        }
    }
    if (opt.passes < 0 || opt.attempts_per_pass < 0 || opt.patience < 0) {
        throw std::runtime_error("--passes/--attempts-per-pass/--patience invalidos.");
    }
    if (opt.boundary_topk <= 0) {
        throw std::runtime_error("--boundary-topk precisa ser > 0.");
    }
    if (opt.push_bisect_iters <= 0) {
        throw std::runtime_error("--push-bisect-iters precisa ser > 0.");
    }
    if (!(opt.push_max_step_frac > 0.0)) {
        throw std::runtime_error("--push-max-step-frac precisa ser > 0.");
    }
    if (opt.plateau_eps < 0.0) {
        throw std::runtime_error("--plateau-eps precisa ser >= 0.");
    }
    if (opt.diag_frac < 0.0) {
        throw std::runtime_error("--diag-frac precisa ser >= 0.");
    }
    if (opt.diag_rand < 0.0) {
        throw std::runtime_error("--diag-rand precisa ser >= 0.");
    }
    if (opt.center_bias < 0.0) {
        throw std::runtime_error("--center-bias precisa ser >= 0.");
    }
    if (opt.interior_prob < 0.0 || opt.interior_prob > 1.0) {
        throw std::runtime_error("--interior-prob precisa estar em [0,1].");
    }
    if (opt.shake_pos < 0.0) {
        throw std::runtime_error("--shake-pos precisa ser >= 0.");
    }
    if (opt.shake_rot_deg < 0.0) {
        throw std::runtime_error("--shake-rot-deg precisa ser >= 0.");
    }
    if (opt.shake_prob < 0.0 || opt.shake_prob > 1.0) {
        throw std::runtime_error("--shake-prob precisa estar em [0,1].");
    }
    if (opt.quantize_decimals < 0) {
        throw std::runtime_error("--quantize-decimals precisa ser >= 0.");
    }

    bool has_target = (opt.target_top > 0) || !opt.target_ns.empty() || opt.target_range_set;
    if (!has_target) {
        throw std::runtime_error("Com --base, especifique --target-top/--target-n/--target-range.");
    }

    return opt;
}

std::vector<char> build_target_mask(const Options& opt,
                                    const Polygon& base_poly,
                                    const SubmissionPoses& base) {
    std::vector<char> target(static_cast<size_t>(opt.n_max + 1), 0);

    if (opt.target_top > 0) {
        struct TermRow {
            int n;
            double term;
        };
        std::vector<TermRow> rows;
        rows.reserve(static_cast<size_t>(opt.n_max));
        for (int n = opt.n_min; n <= opt.n_max; ++n) {
            const auto& poses = base.by_n[static_cast<size_t>(n)];
            if (static_cast<int>(poses.size()) != n) {
                continue;
            }
            const double term = score_term_for_n(base_poly, poses);
            rows.push_back(TermRow{n, term});
        }
        std::sort(rows.begin(), rows.end(), [](const TermRow& a, const TermRow& b) {
            return a.term > b.term;
        });
        const int topk = std::min(opt.target_top, static_cast<int>(rows.size()));
        for (int i = 0; i < topk; ++i) {
            target[static_cast<size_t>(rows[static_cast<size_t>(i)].n)] = 1;
        }
    }

    if (opt.target_range_set) {
        for (int n = opt.target_range_min; n <= opt.target_range_max; ++n) {
            if (n >= opt.n_min && n <= opt.n_max) {
                target[static_cast<size_t>(n)] = 1;
            }
        }
    }

    for (int n : opt.target_ns) {
        if (n >= opt.n_min && n <= opt.n_max) {
            target[static_cast<size_t>(n)] = 1;
        }
    }

    return target;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        SubmissionPoses base = load_submission_poses(opt.base_path, opt.n_max);
        if (static_cast<int>(base.by_n.size()) <= opt.n_max) {
            throw std::runtime_error("Submission incompleta para n_max.");
        }

        Polygon base_poly = get_tree_polygon();
        std::vector<char> target = build_target_mask(opt, base_poly, base);

        std::vector<std::vector<TreePose>> out = base.by_n;
        std::mt19937_64 rng(opt.seed);
        compaction_contact::Params params;
        params.passes = opt.passes;
        params.attempts_per_pass = opt.attempts_per_pass;
        params.patience = opt.patience;
        params.boundary_topk = opt.boundary_topk;
        params.push_bisect_iters = opt.push_bisect_iters;
        params.push_max_step_frac = opt.push_max_step_frac;
        params.plateau_eps = opt.plateau_eps;
        params.alt_axis = opt.alt_axis;
        params.diag_frac = opt.diag_frac;
        params.diag_rand = opt.diag_rand;
        params.center_bias = opt.center_bias;
        params.interior_prob = opt.interior_prob;
        params.shake_pos = opt.shake_pos;
        params.shake_rot_deg = opt.shake_rot_deg;
        params.shake_prob = opt.shake_prob;
        params.final_rigid = opt.final_rigid;
        params.quantize_decimals = opt.quantize_decimals;

        for (int n = opt.n_min; n <= opt.n_max; ++n) {
            if (!target[static_cast<size_t>(n)]) {
                continue;
            }
            auto& poses = out[static_cast<size_t>(n)];
            if (static_cast<int>(poses.size()) != n) {
                continue;
            }

            compaction_contact::Stats res =
                compaction_contact::compact_contact(base_poly, poses, params, rng);
            if (!res.ok) {
                continue;
            }
            const double before_term = (res.side_before * res.side_before) / static_cast<double>(n);
            const double after_term = (res.side_after * res.side_after) / static_cast<double>(n);
            std::cout << "n=" << n
                      << " side " << res.side_before << " -> " << res.side_after
                      << " term " << before_term << " -> " << after_term << "\n";
        }

        std::ofstream out_file(opt.output_path);
        if (!out_file) {
            throw std::runtime_error("Erro ao abrir arquivo de saida: " + opt.output_path);
        }
        out_file << "id,x,y,deg\n";
        for (int n = 1; n <= opt.n_max; ++n) {
            const auto& sol = out[static_cast<size_t>(n)];
            if (static_cast<int>(sol.size()) != n) {
                throw std::runtime_error("Solucao invalida para n=" + std::to_string(n));
            }
            for (int i = 0; i < n; ++i) {
                const auto& pose = sol[static_cast<size_t>(i)];
                out_file << fmt_submission_id(n, i) << ","
                         << fmt_submission_value(pose.x) << ","
                         << fmt_submission_value(pose.y) << ","
                         << fmt_submission_value(pose.deg) << "\n";
            }
        }

        std::cout << "Submission gerada em " << opt.output_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Erro no compact_contact: " << ex.what() << "\n";
        return 1;
    }
}
