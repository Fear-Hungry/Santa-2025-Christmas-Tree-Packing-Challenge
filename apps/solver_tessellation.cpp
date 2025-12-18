#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision.hpp"
#include "ga.hpp"
#include "geom.hpp"
#include "sa.hpp"
#include "submission_io.hpp"
#include "wrap_utils.hpp"

namespace {

constexpr double kSqrt3 = 1.732050807568877293527446341505872366942805254;
constexpr int kOutputDecimals = 9;

struct Options {
    int n_max = 200;
    uint64_t seed = 123456789ULL;
    double spacing_safety = 1.001;
    double shift_a = 0.0;
    double shift_b = 0.0;
    std::vector<double> angle_candidates = {0.0, 15.0, 30.0, 45.0, 60.0, 75.0};
    std::string output_path = "submission_tessellation_cpp.csv";
    bool use_ga = false;
    int ga_pop = 40;
    int ga_gens = 60;
    int ga_elite = 2;
    int ga_tournament = 3;
    double ga_spacing_min = 1.000;
    double ga_spacing_max = 1.010;
    std::vector<double> ga_rot_candidates = {0.0, 180.0};
    int sa_restarts = 0;
    int sa_base_iters = 0;
    int sa_iters_per_n = 0;
    double sa_w_micro = 1.0;
    double sa_w_swap_rot = 0.25;
    double sa_w_relocate = 0.15;
    double sa_w_block_translate = 0.05;
    double sa_w_block_rotate = 0.02;
    double sa_w_lns = 0.001;
    double sa_w_push_contact = 0.0;
    double sa_w_squeeze = 0.0;
    int sa_block_size = 6;
    int sa_lns_remove = 6;
    int sa_hh_segment = 50;
    double sa_hh_reaction = 0.20;
    double sa_overlap_weight = 0.0;
    double sa_overlap_eps_area = 1e-12;
    double sa_overlap_cost_cap = 0.0;
    double sa_plateau_eps = 0.0;
    double sa_w_resolve_overlap = 0.0;
    int sa_resolve_attempts = 6;
    double sa_resolve_step_frac_max = 0.20;
    double sa_resolve_step_frac_min = 0.02;
    double sa_resolve_noise_frac = 0.05;
    double sa_push_max_step_frac = 0.60;
    int sa_push_bisect_iters = 10;
    int sa_squeeze_pushes = 6;
    bool final_rigid = true;
};

struct Candidate {
    TreePose pose;
    double key1;
    double key2;
};

std::pair<Point, Point> hex_basis(double spacing, double angle_deg) {
    Point u{spacing, 0.0};
    Point v{0.5 * spacing, 0.5 * spacing * kSqrt3};
    return {rotate_point(u, angle_deg), rotate_point(v, angle_deg)};
}

double rotation_pattern_deg(int i, int j) {
    (void)i;
    (void)j;
    return 0.0;
}

bool safe_hex_spacing(const Polygon& base_poly,
                      double radius,
                      double spacing,
                      double eps = 1e-9) {
    if (!(spacing > 0.0)) {
        return false;
    }

    auto [u, v] = hex_basis(spacing, 0.0);

    const double limit = 2.0 * radius + eps;
    const double limit_sq = limit * limit;
    int m = static_cast<int>(std::ceil(limit / spacing)) + 2;
    m = std::max(2, m);

    const TreePose origin{0.0, 0.0, rotation_pattern_deg(0, 0)};

    for (int i = -m; i <= m; ++i) {
        for (int j = -m; j <= m; ++j) {
            if (i == 0 && j == 0) {
                continue;
            }
            double dx = i * u.x + j * v.x;
            double dy = i * u.y + j * v.y;
            double d2 = dx * dx + dy * dy;
            if (d2 > limit_sq) {
                continue;
            }

            const TreePose other{dx, dy, rotation_pattern_deg(i, j)};
            const std::vector<TreePose> poses{origin, other};
            if (any_overlap(base_poly, poses, radius, eps)) {
                return false;
            }
        }
    }
    return true;
}

double find_min_safe_hex_spacing(const Polygon& base_poly,
                                double radius,
                                double eps = 1e-9) {
    double lo = 0.0;
    double hi = 2.0 * radius;
    if (!safe_hex_spacing(base_poly, radius, hi, eps)) {
        // Pela definição de radius como círculo envolvente, hi deveria ser
        // sempre seguro. Mantemos um fallback por robustez.
        for (int it = 0; it < 30 && !safe_hex_spacing(base_poly, radius, hi, eps);
             ++it) {
            hi *= 1.5;
        }
    }

    for (int it = 0; it < 70; ++it) {
        double mid = 0.5 * (lo + hi);
        if (safe_hex_spacing(base_poly, radius, mid, eps)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
}

std::vector<TreePose> generate_hex_lattice_poses(int n,
                                                 double spacing,
                                                 double angle_deg,
                                                 double shift_a,
                                                 double shift_b) {
    if (n <= 0) {
        return {};
    }

    auto [u, v] = hex_basis(spacing, angle_deg);

    int m = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))) + 8;

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>((2 * m + 1) * (2 * m + 1)));
    for (int i = -m; i <= m; ++i) {
        for (int j = -m; j <= m; ++j) {
            double ci = static_cast<double>(i) - shift_a;
            double cj = static_cast<double>(j) - shift_b;
            double x = ci * u.x + cj * v.x;
            double y = ci * u.y + cj * v.y;
            double key1 = std::max(std::abs(x), std::abs(y));
            double key2 = std::hypot(x, y);
            candidates.push_back({TreePose{x, y, rotation_pattern_deg(i, j)},
                                  key1,
                                  key2});
        }
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.key1 != b.key1) {
                      return a.key1 < b.key1;
                  }
                  if (a.key2 != b.key2) {
                      return a.key2 < b.key2;
                  }
                  if (a.pose.x != b.pose.x) {
                      return a.pose.x < b.pose.x;
                  }
                  if (a.pose.y != b.pose.y) {
                      return a.pose.y < b.pose.y;
                  }
                  return a.pose.deg < b.pose.deg;
              });

    std::vector<TreePose> out;
    out.reserve(static_cast<size_t>(n));
    for (const auto& c : candidates) {
        out.push_back(c.pose);
        if (static_cast<int>(out.size()) >= n) {
            break;
        }
    }
    return out;
}

double local_score_for_n(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    auto polys = transformed_polygons(base_poly, poses);
    double s = bounding_square_side(polys);
    return (s * s) / static_cast<double>(poses.size());
}

Options parse_args(int argc, char** argv) {
    Options opt;

    auto parse_int = [](const std::string& s) -> int {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Inteiro inválido: " + s);
        }
        return v;
    };
    auto parse_u64 = [](const std::string& s) -> uint64_t {
        size_t pos = 0;
        uint64_t v = std::stoull(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("uint64 inválido: " + s);
        }
        return v;
    };
    auto parse_double = [](const std::string& s) -> double {
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Double inválido: " + s);
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

        if (arg == "--n-max") {
            opt.n_max = parse_int(need(arg));
        } else if (arg == "--seed") {
            opt.seed = parse_u64(need(arg));
        } else if (arg == "--spacing-safety") {
            opt.spacing_safety = parse_double(need(arg));
        } else if (arg == "--use-ga") {
            opt.use_ga = true;
        } else if (arg == "--ga-pop") {
            opt.ga_pop = parse_int(need(arg));
        } else if (arg == "--ga-gens") {
            opt.ga_gens = parse_int(need(arg));
        } else if (arg == "--ga-elite") {
            opt.ga_elite = parse_int(need(arg));
        } else if (arg == "--ga-tournament") {
            opt.ga_tournament = parse_int(need(arg));
        } else if (arg == "--ga-spacing-min") {
            opt.ga_spacing_min = parse_double(need(arg));
        } else if (arg == "--ga-spacing-max") {
            opt.ga_spacing_max = parse_double(need(arg));
        } else if (arg == "--ga-rots") {
            std::string s = need(arg);
            opt.ga_rot_candidates.clear();
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) {
                    continue;
                }
                opt.ga_rot_candidates.push_back(parse_double(item));
            }
            if (opt.ga_rot_candidates.empty()) {
                throw std::runtime_error("--ga-rots vazio.");
            }
        } else if (arg == "--shift-a") {
            opt.shift_a = parse_double(need(arg));
        } else if (arg == "--shift-b") {
            opt.shift_b = parse_double(need(arg));
        } else if (arg == "--shift") {
            std::string s = need(arg);
            std::stringstream ss(s);
            std::string a, b;
            if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',')) {
                throw std::runtime_error("--shift precisa ser 'a,b'.");
            }
            opt.shift_a = parse_double(a);
            opt.shift_b = parse_double(b);
        } else if (arg == "--angles") {
            std::string s = need(arg);
            opt.angle_candidates.clear();
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) {
                    continue;
                }
                opt.angle_candidates.push_back(parse_double(item));
            }
            if (opt.angle_candidates.empty()) {
                throw std::runtime_error("--angles vazio.");
            }
        } else if (arg == "--output") {
            opt.output_path = need(arg);
        } else if (arg == "--sa-restarts") {
            opt.sa_restarts = parse_int(need(arg));
        } else if (arg == "--sa-base-iters") {
            opt.sa_base_iters = parse_int(need(arg));
        } else if (arg == "--sa-iters-per-n") {
            opt.sa_iters_per_n = parse_int(need(arg));
        } else if (arg == "--sa-w-micro") {
            opt.sa_w_micro = parse_double(need(arg));
        } else if (arg == "--sa-w-swap-rot") {
            opt.sa_w_swap_rot = parse_double(need(arg));
        } else if (arg == "--sa-w-relocate") {
            opt.sa_w_relocate = parse_double(need(arg));
        } else if (arg == "--sa-w-block-translate") {
            opt.sa_w_block_translate = parse_double(need(arg));
        } else if (arg == "--sa-w-block-rotate") {
            opt.sa_w_block_rotate = parse_double(need(arg));
        } else if (arg == "--sa-w-lns") {
            opt.sa_w_lns = parse_double(need(arg));
        } else if (arg == "--sa-w-push-contact") {
            opt.sa_w_push_contact = parse_double(need(arg));
        } else if (arg == "--sa-w-squeeze") {
            opt.sa_w_squeeze = parse_double(need(arg));
        } else if (arg == "--sa-block-size") {
            opt.sa_block_size = parse_int(need(arg));
        } else if (arg == "--sa-lns-remove") {
            opt.sa_lns_remove = parse_int(need(arg));
        } else if (arg == "--sa-hh-segment") {
            opt.sa_hh_segment = parse_int(need(arg));
        } else if (arg == "--sa-hh-reaction") {
            opt.sa_hh_reaction = parse_double(need(arg));
        } else if (arg == "--sa-overlap-weight") {
            opt.sa_overlap_weight = parse_double(need(arg));
        } else if (arg == "--sa-overlap-eps-area") {
            opt.sa_overlap_eps_area = parse_double(need(arg));
        } else if (arg == "--sa-overlap-cost-cap") {
            opt.sa_overlap_cost_cap = parse_double(need(arg));
        } else if (arg == "--sa-plateau-eps") {
            opt.sa_plateau_eps = parse_double(need(arg));
        } else if (arg == "--sa-w-resolve-overlap") {
            opt.sa_w_resolve_overlap = parse_double(need(arg));
        } else if (arg == "--sa-resolve-attempts") {
            opt.sa_resolve_attempts = parse_int(need(arg));
        } else if (arg == "--sa-resolve-step-frac-max") {
            opt.sa_resolve_step_frac_max = parse_double(need(arg));
        } else if (arg == "--sa-resolve-step-frac-min") {
            opt.sa_resolve_step_frac_min = parse_double(need(arg));
        } else if (arg == "--sa-resolve-noise-frac") {
            opt.sa_resolve_noise_frac = parse_double(need(arg));
        } else if (arg == "--sa-push-max-step-frac") {
            opt.sa_push_max_step_frac = parse_double(need(arg));
        } else if (arg == "--sa-push-bisect-iters") {
            opt.sa_push_bisect_iters = parse_int(need(arg));
        } else if (arg == "--sa-squeeze-pushes") {
            opt.sa_squeeze_pushes = parse_int(need(arg));
        } else if (arg == "--no-final-rigid" || arg == "--no-sa-rigid") {
            opt.final_rigid = false;
        } else {
            throw std::runtime_error("Argumento desconhecido: " + arg);
        }
    }

    if (opt.n_max <= 0 || opt.n_max > 200) {
        throw std::runtime_error("--n-max precisa estar em [1, 200].");
    }
    if (!(opt.spacing_safety >= 1.0)) {
        throw std::runtime_error("--spacing-safety precisa ser >= 1.0.");
    }
    if (opt.ga_pop <= 0) {
        throw std::runtime_error("--ga-pop precisa ser > 0.");
    }
    if (opt.ga_gens < 0) {
        throw std::runtime_error("--ga-gens precisa ser >= 0.");
    }
    if (opt.ga_elite < 0 || opt.ga_elite > opt.ga_pop) {
        throw std::runtime_error("--ga-elite precisa estar em [0, ga-pop].");
    }
    if (opt.ga_tournament <= 0) {
        throw std::runtime_error("--ga-tournament precisa ser > 0.");
    }
    if (!(opt.ga_spacing_min >= 1.0) || !(opt.ga_spacing_max >= opt.ga_spacing_min)) {
        throw std::runtime_error("--ga-spacing-min/max inválidos (precisa min>=1 e max>=min).");
    }
    if (opt.sa_restarts < 0 || opt.sa_base_iters < 0 || opt.sa_iters_per_n < 0) {
        throw std::runtime_error("Parâmetros de SA precisam ser >= 0.");
    }
    if (opt.sa_w_micro < 0.0 || opt.sa_w_swap_rot < 0.0 || opt.sa_w_relocate < 0.0 ||
        opt.sa_w_block_translate < 0.0 || opt.sa_w_block_rotate < 0.0 || opt.sa_w_lns < 0.0 ||
        opt.sa_w_resolve_overlap < 0.0 || opt.sa_w_push_contact < 0.0 || opt.sa_w_squeeze < 0.0) {
        throw std::runtime_error("Pesos de SA precisam ser >= 0.");
    }
    if (opt.sa_block_size <= 0) {
        throw std::runtime_error("--sa-block-size precisa ser > 0.");
    }
    if (opt.sa_lns_remove < 0) {
        throw std::runtime_error("--sa-lns-remove precisa ser >= 0.");
    }
    if (opt.sa_hh_segment < 0) {
        throw std::runtime_error("--sa-hh-segment precisa ser >= 0.");
    }
    if (opt.sa_hh_reaction < 0.0 || opt.sa_hh_reaction > 1.0) {
        throw std::runtime_error("--sa-hh-reaction precisa estar em [0, 1].");
    }
    if (!(opt.sa_overlap_weight >= 0.0)) {
        throw std::runtime_error("--sa-overlap-weight precisa ser >= 0.");
    }
    if (!(opt.sa_overlap_eps_area >= 0.0)) {
        throw std::runtime_error("--sa-overlap-eps-area precisa ser >= 0.");
    }
    if (!(opt.sa_overlap_cost_cap >= 0.0)) {
        throw std::runtime_error("--sa-overlap-cost-cap precisa ser >= 0.");
    }
    if (!(opt.sa_plateau_eps >= 0.0)) {
        throw std::runtime_error("--sa-plateau-eps precisa ser >= 0.");
    }
    if (opt.sa_resolve_attempts <= 0) {
        throw std::runtime_error("--sa-resolve-attempts precisa ser > 0.");
    }
    if (!(opt.sa_resolve_step_frac_max > 0.0) || !(opt.sa_resolve_step_frac_min > 0.0) ||
        opt.sa_resolve_step_frac_min > opt.sa_resolve_step_frac_max) {
        throw std::runtime_error("--sa-resolve-step-frac-min/max inválidos.");
    }
    if (!(opt.sa_resolve_noise_frac >= 0.0)) {
        throw std::runtime_error("--sa-resolve-noise-frac precisa ser >= 0.");
    }
    if (!(opt.sa_push_max_step_frac > 0.0)) {
        throw std::runtime_error("--sa-push-max-step-frac precisa ser > 0.");
    }
    if (opt.sa_push_bisect_iters <= 0) {
        throw std::runtime_error("--sa-push-bisect-iters precisa ser > 0.");
    }
    if (opt.sa_squeeze_pushes < 0) {
        throw std::runtime_error("--sa-squeeze-pushes precisa ser >= 0.");
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);
        const double shift_a = wrap01(opt.shift_a);
        const double shift_b = wrap01(opt.shift_b);

        const double min_spacing = find_min_safe_hex_spacing(base_poly, radius);
        const double spacing = min_spacing * opt.spacing_safety;

        std::ofstream out(opt.output_path);
        if (!out) {
            throw std::runtime_error("Erro ao abrir arquivo de saída: " + opt.output_path);
        }

        out << "id,x,y,deg\n";

        SARefiner sa(base_poly, radius);
        double total_score = 0.0;

        std::vector<TreePose> ga_poses_sorted;
        double ga_side = std::numeric_limits<double>::infinity();
        if (opt.use_ga) {
            GAParams gp;
            gp.pop_size = opt.ga_pop;
            gp.generations = opt.ga_gens;
            gp.elite = opt.ga_elite;
            gp.tournament_k = opt.ga_tournament;
            gp.spacing_safety_min = opt.ga_spacing_min;
            gp.spacing_safety_max = opt.ga_spacing_max;
            gp.lattice_angle_candidates = opt.angle_candidates;
            gp.rotation_candidates = opt.ga_rot_candidates;

            GlobalSearchGA ga(base_poly, radius);
            GAResult gr = ga.solve(
                opt.n_max,
                opt.seed ^ 0xA7F4C3B2D1E0F987ULL,
                gp);

            if (!gr.best_poses.empty() && std::isfinite(gr.best_side)) {
                auto q = quantize_poses(gr.best_poses);
                if (!any_overlap(base_poly, q, radius)) {
                    ga_poses_sorted = std::move(q);
                    ga_side = gr.best_side;
                    std::sort(ga_poses_sorted.begin(),
                              ga_poses_sorted.end(),
                              [](const TreePose& a, const TreePose& b) {
                                  double a1 = std::max(std::abs(a.x), std::abs(a.y));
                                  double b1 = std::max(std::abs(b.x), std::abs(b.y));
                                  if (a1 != b1) {
                                      return a1 < b1;
                                  }
                                  double a2 = std::hypot(a.x, a.y);
                                  double b2 = std::hypot(b.x, b.y);
                                  if (a2 != b2) {
                                      return a2 < b2;
                                  }
                                  if (a.x != b.x) {
                                      return a.x < b.x;
                                  }
                                  if (a.y != b.y) {
                                      return a.y < b.y;
                                  }
                                  return a.deg < b.deg;
                              });

                    std::cout << "GA (n_max=" << opt.n_max << ") best_side: "
                              << std::fixed << std::setprecision(9) << ga_side
                              << " (spacing=" << gr.best_spacing
                              << ", angle=" << gr.best_angle_deg
                              << ", shift=" << gr.best_shift_a << "," << gr.best_shift_b
                              << ")\n";
                } else {
                    std::cerr << "Aviso: GA produziu overlap após quantização; ignorando GA.\n";
                }
            } else {
                std::cerr << "Aviso: GA não encontrou solução; ignorando GA.\n";
            }
        }

        for (int n = 1; n <= opt.n_max; ++n) {
            double best_score_n = std::numeric_limits<double>::infinity();
            std::vector<TreePose> best_poses;

            for (double angle_deg : opt.angle_candidates) {
                auto poses = generate_hex_lattice_poses(n, spacing, angle_deg, shift_a, shift_b);
                if (any_overlap(base_poly, poses, radius)) {
                    continue;
                }
                auto poses_q = quantize_poses(poses);
                if (any_overlap(base_poly, poses_q, radius)) {
                    continue;
                }
                double score_n = local_score_for_n(base_poly, poses_q);
                if (score_n < best_score_n) {
                    best_score_n = score_n;
                    best_poses = std::move(poses_q);
                }
            }

            if (!ga_poses_sorted.empty() && static_cast<int>(ga_poses_sorted.size()) >= n) {
                std::vector<TreePose> ga_prefix(ga_poses_sorted.begin(), ga_poses_sorted.begin() + n);
                double score_n = local_score_for_n(base_poly, ga_prefix);
                if (score_n + 1e-15 < best_score_n) {
                    best_score_n = score_n;
                    best_poses = std::move(ga_prefix);
                }
            }

            if (!std::isfinite(best_score_n)) {
                throw std::runtime_error(
                    "Não foi possível gerar poses válidas para n=" +
                    std::to_string(n) + ".");
            }

            if (opt.sa_restarts > 0 && opt.sa_base_iters > 0) {
                SARefiner::Params p;
                p.iters = opt.sa_base_iters + opt.sa_iters_per_n * n;
                p.w_micro = opt.sa_w_micro;
                p.w_swap_rot = opt.sa_w_swap_rot;
                p.w_relocate = opt.sa_w_relocate;
                p.w_block_translate = opt.sa_w_block_translate;
                p.w_block_rotate = opt.sa_w_block_rotate;
                p.w_lns = opt.sa_w_lns;
                p.w_push_contact = opt.sa_w_push_contact;
                p.w_squeeze = opt.sa_w_squeeze;
                p.block_size = opt.sa_block_size;
                p.lns_remove = opt.sa_lns_remove;
                p.hh_segment = opt.sa_hh_segment;
                p.hh_reaction = opt.sa_hh_reaction;
                p.overlap_weight = opt.sa_overlap_weight;
                p.overlap_eps_area = opt.sa_overlap_eps_area;
                p.overlap_cost_cap = opt.sa_overlap_cost_cap;
                p.plateau_eps = opt.sa_plateau_eps;
                p.w_resolve_overlap = opt.sa_w_resolve_overlap;
                p.resolve_attempts = opt.sa_resolve_attempts;
                p.resolve_step_frac_max = opt.sa_resolve_step_frac_max;
                p.resolve_step_frac_min = opt.sa_resolve_step_frac_min;
                p.resolve_noise_frac = opt.sa_resolve_noise_frac;
                p.push_max_step_frac = opt.sa_push_max_step_frac;
                p.push_bisect_iters = opt.sa_push_bisect_iters;
                p.squeeze_pushes = opt.sa_squeeze_pushes;

                for (int r = 0; r < opt.sa_restarts; ++r) {
                    uint64_t seed =
                        opt.seed ^
                        (0x9e3779b97f4a7c15ULL +
                         static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
                         static_cast<uint64_t>(r) * 0x94d049bb133111ebULL);
                    SARefiner::Result res = sa.refine_min_side(best_poses, seed, p);
                    auto cand_q = quantize_poses(res.best_poses);
                    if (any_overlap(base_poly, cand_q, radius)) {
                        continue;
                    }
                    double cand_score_n = local_score_for_n(base_poly, cand_q);
                    if (cand_score_n + 1e-15 < best_score_n) {
                        best_score_n = cand_score_n;
                        best_poses = std::move(cand_q);
                    }
                }
            }

            if (opt.final_rigid) {
                std::vector<TreePose> rigid_sol = best_poses;
                optimize_rigid_rotation(base_poly, rigid_sol);
                auto rigid_q = quantize_poses(rigid_sol);
                if (!any_overlap(base_poly, rigid_q, radius)) {
                    double rigid_score_n = local_score_for_n(base_poly, rigid_q);
                    if (rigid_score_n + 1e-15 < best_score_n) {
                        best_score_n = rigid_score_n;
                        best_poses = std::move(rigid_q);
                    }
                }
            }

            total_score += best_score_n;

            for (int i = 0; i < n; ++i) {
                const auto& pose = best_poses[i];
                out << std::setw(3) << std::setfill('0') << n << "_" << i << ","
                    << fmt_submission_value(pose.x) << ","
                    << fmt_submission_value(pose.y) << ","
                    << fmt_submission_value(pose.deg) << "\n";
            }
        }

        std::cout << "Submission gerada em " << opt.output_path << "\n";
        std::cout << "Score (local): " << std::fixed << std::setprecision(9)
                  << total_score << "\n";
        std::cout << "Hex spacing: " << std::fixed << std::setprecision(9)
                  << spacing << "\n";
        std::cout << "Angles: " << opt.angle_candidates.size() << "\n";
        std::cout << "SA restarts: " << opt.sa_restarts << "\n";
        std::cout << "SA base iters: " << opt.sa_base_iters << "\n";
        std::cout << "SA iters per n: " << opt.sa_iters_per_n << "\n";
        std::cout << "Final rigid: " << (opt.final_rigid ? "on" : "off") << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
