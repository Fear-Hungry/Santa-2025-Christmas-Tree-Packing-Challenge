#include "ensemble_submissions_cli.hpp"

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
#include "micro_adjust.hpp"
#include "submission_io.hpp"

namespace {

constexpr int kNMax = 200;
constexpr long double kEvalEps = 1e-18L;
constexpr double kSideEps = 1e-15;

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

struct RemovalResult {
    std::vector<TreePose> poses;
    double side = std::numeric_limits<double>::infinity();
    int removed_idx = -1;
};

std::vector<TreePose> remove_pose_at(const std::vector<TreePose>& poses, int drop_idx) {
    std::vector<TreePose> out;
    out.reserve(poses.size() - 1);
    for (size_t i = 0; i < poses.size(); ++i) {
        if (static_cast<int>(i) == drop_idx) {
            continue;
        }
        out.push_back(poses[i]);
    }
    return out;
}

RemovalResult best_removal_solution(const Polygon& base_poly,
                                    const std::vector<TreePose>& poses,
                                    double side_eps) {
    if (poses.size() < 2) {
        throw std::runtime_error("cross-check: precisa de n >= 2 para remover.");
    }
    RemovalResult best;
    for (int i = 0; i < static_cast<int>(poses.size()); ++i) {
        std::vector<TreePose> cand = remove_pose_at(poses, i);
        double side = side_for_solution(base_poly, cand);
        if (side + side_eps < best.side ||
            (std::abs(side - best.side) <= side_eps && i < best.removed_idx)) {
            best.side = side;
            best.removed_idx = i;
            best.poses = std::move(cand);
        }
    }
    return best;
}

struct Options {
    std::string output_path;
    std::vector<std::string> inputs;
    bool final_rigid = true;
    bool cross_check_n = false;
    double micro_rot_eps = 0.0;
    int micro_rot_steps = 0;
    double micro_shift_eps = 0.0;
    int micro_shift_steps = 0;
    int debug_n = -1;
    int debug_top = 5;
};

void print_usage(const char* exe) {
    std::cerr << "Uso: " << exe
              << " output.csv input1.csv [input2.csv ...] [--no-final-rigid]"
                 " [--debug-n N] [--debug-top K] [--cross-check-n]"
                 " [--micro-rot-eps DEG] [--micro-rot-steps K]"
                 " [--micro-shift-eps DELTA] [--micro-shift-steps K]\n";
}

int parse_int(const std::string& s, const std::string& name) {
    size_t pos = 0;
    int v = std::stoi(s, &pos);
    if (pos != s.size()) {
        throw std::runtime_error("Inteiro inválido para " + name + ": " + s);
    }
    return v;
}

double parse_double(const std::string& s, const std::string& name) {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) {
        throw std::runtime_error("Double inválido para " + name + ": " + s);
    }
    return v;
}

bool is_disable_final_rigid_flag(const std::string& arg) {
    return (arg == "--no-final-rigid" || arg == "--no-sa-rigid");
}

struct OptionParser {
    int argc = 0;
    char** argv = nullptr;
    Options opt;

    OptionParser(int argc_in, char** argv_in) : argc(argc_in), argv(argv_in) {
        opt.output_path = argv[1];
    }

    Options parse() {
        for (int i = 2; i < argc; ++i) {
            parse_one(i);
        }
        validate();
        return std::move(opt);
    }

    std::string need(const std::string& name, int& i) const {
        if (i + 1 >= argc) {
            throw std::runtime_error("Faltou valor para " + name);
        }
        return argv[++i];
    }

    void parse_one(int& i) {
        std::string arg = argv[i];
        if (parse_bool_flag(arg)) {
            return;
        }
        if (parse_micro_flag(arg, i)) {
            return;
        }
        if (parse_debug_flag(arg, i)) {
            return;
        }
        opt.inputs.push_back(std::move(arg));
    }

    bool parse_bool_flag(const std::string& arg) {
        if (is_disable_final_rigid_flag(arg)) {
            opt.final_rigid = false;
            return true;
        }
        if (arg == "--cross-check-n") {
            opt.cross_check_n = true;
            return true;
        }
        return false;
    }

    bool parse_micro_flag(const std::string& arg, int& i) {
        if (arg == "--micro-rot-eps") {
            opt.micro_rot_eps = parse_double(need(arg, i), arg);
            return true;
        }
        if (arg == "--micro-rot-steps") {
            opt.micro_rot_steps = parse_int(need(arg, i), arg);
            return true;
        }
        if (arg == "--micro-shift-eps") {
            opt.micro_shift_eps = parse_double(need(arg, i), arg);
            return true;
        }
        if (arg == "--micro-shift-steps") {
            opt.micro_shift_steps = parse_int(need(arg, i), arg);
            return true;
        }
        return false;
    }

    bool parse_debug_flag(const std::string& arg, int& i) {
        if (arg == "--debug-n") {
            opt.debug_n = parse_int(need(arg, i), arg);
            return true;
        }
        if (arg == "--debug-top") {
            opt.debug_top = parse_int(need(arg, i), arg);
            return true;
        }
        return false;
    }

    void validate() const {
        validate_inputs();
        validate_debug();
        validate_micro();
    }

    void validate_inputs() const {
        if (opt.inputs.empty()) {
            throw std::runtime_error("Precisa de ao menos 1 input.");
        }
    }

    void validate_debug() const {
        if (opt.debug_n == -1) {
            return;
        }
        if (opt.debug_n < 1 || opt.debug_n > kNMax) {
            throw std::runtime_error("--debug-n precisa estar em [1,200] ou -1.");
        }
        if (opt.debug_top <= 0) {
            throw std::runtime_error("--debug-top precisa ser > 0.");
        }
    }

    void validate_micro() const {
        if (opt.micro_rot_steps < 0 || opt.micro_shift_steps < 0) {
            throw std::runtime_error("--micro-rot-steps/--micro-shift-steps precisam ser >= 0.");
        }
        if (opt.micro_rot_eps < 0.0 || opt.micro_shift_eps < 0.0) {
            throw std::runtime_error("--micro-rot-eps/--micro-shift-eps precisam ser >= 0.");
        }
        if (opt.micro_rot_steps > 0 && !(opt.micro_rot_eps > 0.0)) {
            throw std::runtime_error("--micro-rot-eps precisa ser > 0 quando steps > 0.");
        }
        if (opt.micro_shift_steps > 0 && !(opt.micro_shift_eps > 0.0)) {
            throw std::runtime_error("--micro-shift-eps precisa ser > 0 quando steps > 0.");
        }
        if (opt.debug_top <= 0) {
            throw std::runtime_error("--debug-top precisa ser > 0.");
        }
    }
};

Options parse_options(int argc, char** argv) {
    OptionParser parser(argc, argv);
    return parser.parse();
}

std::vector<SubmissionPoses> load_inputs(const std::vector<std::string>& paths) {
    std::vector<SubmissionPoses> subs;
    subs.reserve(paths.size());
    for (const auto& p : paths) {
        subs.push_back(load_submission_poses(p, kNMax));
    }
    return subs;
}

struct CandidateEval {
    bool ok = false;
    double pre_side = 0.0;
    long double pre_term = 0.0L;
    long double eval_term = 0.0L;
};

struct GeometryContext {
    const Polygon& base_poly;
    double radius = 0.0;
};

struct EvalContext {
    GeometryContext geom;
    bool final_rigid = true;
};

CandidateEval evaluate_candidate(const EvalContext& ctx,
                                 int n,
                                 const std::vector<TreePose>& poses) {
    CandidateEval out;
    if (any_overlap(ctx.geom.base_poly, poses, ctx.geom.radius)) {
        return out;
    }
    out.ok = true;
    out.pre_side = side_for_solution(ctx.geom.base_poly, poses);
    out.pre_term = term_for_side(out.pre_side, n);
    out.eval_term = out.pre_term;

    if (!ctx.final_rigid) {
        return out;
    }

    std::vector<TreePose> rigid_sol = poses;
    optimize_rigid_rotation(ctx.geom.base_poly, rigid_sol);
    auto rigid_q = quantize_poses(rigid_sol);
    if (any_overlap(ctx.geom.base_poly, rigid_q, ctx.geom.radius)) {
        return out;
    }

    double rigid_side = side_for_solution(ctx.geom.base_poly, rigid_q);
    if (rigid_side + kSideEps < out.pre_side) {
        out.eval_term = term_for_side(rigid_side, n);
    }
    return out;
}

struct SelectionResult {
    int best_k = -1;
    long double best_eval_term = std::numeric_limits<long double>::infinity();
    std::vector<TreePose> best_poses;
    std::vector<DebugEntry> debug_entries;
};

bool is_better_eval_term(long double cand_term,
                         int cand_k,
                         long double best_term,
                         int best_k) {
    if (cand_term + kEvalEps < best_term) {
        return true;
    }
    if (std::fabsl(cand_term - best_term) > kEvalEps) {
        return false;
    }
    return cand_k < best_k;
}

SelectionResult select_best_for_n(const EvalContext& ctx,
                                  const std::vector<SubmissionPoses>& subs,
                                  int n,
                                  bool debug_this_n) {
    SelectionResult out;
    if (debug_this_n) {
        out.debug_entries.reserve(subs.size());
    }

    for (int k = 0; k < static_cast<int>(subs.size()); ++k) {
        const auto& poses = subs[static_cast<size_t>(k)].by_n[static_cast<size_t>(n)];
        CandidateEval eval = evaluate_candidate(ctx, n, poses);
        if (!eval.ok) {
            continue;
        }
        if (debug_this_n) {
            out.debug_entries.push_back(DebugEntry{k, eval.pre_term, eval.eval_term});
        }

        if (is_better_eval_term(eval.eval_term, k, out.best_eval_term, out.best_k)) {
            out.best_eval_term = eval.eval_term;
            out.best_k = k;
            out.best_poses = poses;
        }
    }

    if (out.best_k < 0) {
        throw std::runtime_error("Nenhum input válido para n=" + std::to_string(n));
    }
    return out;
}

struct DebugContext {
    int n = 0;
    int debug_top = 5;
    const std::vector<std::string>& inputs;
};

void print_debug_for_n(const DebugContext& ctx,
                       const std::vector<DebugEntry>& entries,
                       int best_k) {
    auto print_top = [&](const char* label,
                         const std::vector<DebugEntry>& es,
                         bool use_post) {
        const int k = std::min(ctx.debug_top, static_cast<int>(es.size()));
        std::cerr << "[debug] n=" << ctx.n << " top-" << k << " " << label << "\n";
        std::cerr << "rank,k,term,input\n";
        for (int i = 0; i < k; ++i) {
            const auto& e = es[static_cast<size_t>(i)];
            long double t = use_post ? e.post_term : e.pre_term;
            std::cerr << (i + 1) << ","
                      << e.k << ","
                      << std::fixed << std::setprecision(12)
                      << static_cast<double>(t) << ","
                      << ctx.inputs[static_cast<size_t>(e.k)] << "\n";
        }
    };

    auto by_pre = entries;
    auto by_post = entries;
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

    auto chosen_it = std::find_if(entries.begin(),
                                  entries.end(),
                                  [&](const DebugEntry& e) { return e.k == best_k; });
    if (chosen_it == entries.end()) {
        return;
    }

    std::cerr << "[debug] chosen k=" << best_k
              << " pre_term=" << std::fixed << std::setprecision(12)
              << static_cast<double>(chosen_it->pre_term)
              << " post_term=" << static_cast<double>(chosen_it->post_term)
              << " input=" << ctx.inputs[static_cast<size_t>(best_k)]
              << "\n";
}

void apply_final_rigid_if_better(const GeometryContext& ctx,
                                 std::vector<TreePose>& poses,
                                 double& side) {
    std::vector<TreePose> rigid_sol = poses;
    optimize_rigid_rotation(ctx.base_poly, rigid_sol);
    auto rigid_q = quantize_poses(rigid_sol);
    if (any_overlap(ctx.base_poly, rigid_q, ctx.radius)) {
        return;
    }
    double rigid_side = side_for_solution(ctx.base_poly, rigid_q);
    if (rigid_side + kSideEps < side) {
        side = rigid_side;
        poses = std::move(rigid_q);
    }
}

bool apply_micro_adjust_if_better(const GeometryContext& ctx,
                                  const MicroAdjustOptions& micro_opt,
                                  std::vector<TreePose>& poses,
                                  double& side) {
    MicroAdjustOutcome micro_out =
        apply_micro_adjustments(ctx.base_poly, poses, ctx.radius, micro_opt);
    if (!micro_out.result.improved) {
        return false;
    }
    if (!(micro_out.result.best_side + kSideEps < side)) {
        return false;
    }
    side = micro_out.result.best_side;
    poses = std::move(micro_out.poses);
    return true;
}

struct BuiltSolution {
    std::vector<TreePose> poses;
    double side = std::numeric_limits<double>::infinity();
    bool micro_improved = false;
};

struct PostprocessContext {
    GeometryContext geom;
    bool final_rigid = true;
    bool use_micro = false;
    MicroAdjustOptions micro_opt;
};

BuiltSolution build_solution(const PostprocessContext& ctx,
                             std::vector<TreePose> poses) {
    BuiltSolution out;
    out.poses = std::move(poses);
    out.side = side_for_solution(ctx.geom.base_poly, out.poses);

    if (ctx.final_rigid) {
        apply_final_rigid_if_better(ctx.geom, out.poses, out.side);
    }
    if (ctx.use_micro) {
        out.micro_improved = apply_micro_adjust_if_better(
            ctx.geom, ctx.micro_opt, out.poses, out.side);
    }
    return out;
}

int apply_cross_check(const Polygon& base_poly,
                      std::vector<std::vector<TreePose>>& out_by_n,
                      std::vector<double>& side_by_n) {
    int fixes = 0;
    for (int n = kNMax - 1; n >= 1; --n) {
        if (side_by_n[static_cast<size_t>(n)] <= side_by_n[static_cast<size_t>(n + 1)] + kSideEps) {
            continue;
        }
        const auto& next_sol = out_by_n[static_cast<size_t>(n + 1)];
        RemovalResult best = best_removal_solution(base_poly, next_sol, kSideEps);
        out_by_n[static_cast<size_t>(n)] = std::move(best.poses);
        side_by_n[static_cast<size_t>(n)] = best.side;
        ++fixes;
    }
    return fixes;
}

double write_output(const std::string& output_path,
                    const std::vector<std::vector<TreePose>>& out_by_n,
                    const std::vector<double>& side_by_n) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Erro ao abrir arquivo de saída: " + output_path);
    }
    out << "id,x,y,deg\n";

    double total_score = 0.0;
    for (int n = 1; n <= kNMax; ++n) {
        double out_side = side_by_n[static_cast<size_t>(n)];
        total_score += (out_side * out_side) / static_cast<double>(n);
        const auto& sol = out_by_n[static_cast<size_t>(n)];
        if (static_cast<int>(sol.size()) != n) {
            throw std::runtime_error("Tamanho inválido para n=" + std::to_string(n));
        }
        for (int i = 0; i < n; ++i) {
            const auto& p = sol[static_cast<size_t>(i)];
            out << std::setw(3) << std::setfill('0') << n << "_" << i << ","
                << fmt_submission_value(p.x) << ","
                << fmt_submission_value(p.y) << ","
                << fmt_submission_value(p.deg) << "\n";
        }
    }
    return total_score;
}

struct RunStats {
    bool use_micro = false;
    int micro_improved = 0;
    int cross_fixes = 0;
    double total_score = 0.0;
};

void print_summary(const Options& opt,
                   const RunStats& stats) {
    std::cout << "Submission ensembling gerada em " << opt.output_path << "\n";
    std::cout << "Inputs: " << opt.inputs.size() << "\n";
    std::cout << "Score (local): " << std::fixed << std::setprecision(9)
              << stats.total_score << "\n";
    std::cout << "Final rigid: " << (opt.final_rigid ? "on" : "off") << "\n";
    std::cout << "Micro adjust: " << (stats.use_micro ? "on" : "off");
    if (stats.use_micro) {
        std::cout << " (rot_eps=" << opt.micro_rot_eps
                  << ", rot_steps=" << opt.micro_rot_steps
                  << ", shift_eps=" << opt.micro_shift_eps
                  << ", shift_steps=" << opt.micro_shift_steps
                  << ", improved=" << stats.micro_improved << ")";
    }
    std::cout << "\n";
    std::cout << "Cross-check N: " << (opt.cross_check_n ? "on" : "off");
    if (opt.cross_check_n) {
        std::cout << " (fixes=" << stats.cross_fixes << ")";
    }
    std::cout << "\n";
}

}  // namespace

int ensemble_submissions_cli(int argc, char** argv) {
    try {
        if (argc < 3) {
            print_usage(argv[0]);
            return 2;
        }

        Options opt = parse_options(argc, argv);
        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);
        std::vector<SubmissionPoses> subs = load_inputs(opt.inputs);

        std::vector<std::vector<TreePose>> out_by_n(201);
        std::vector<double> side_by_n(201, std::numeric_limits<double>::infinity());

        RunStats stats;
        stats.use_micro =
            (opt.micro_rot_steps > 0 && opt.micro_rot_eps > 0.0) ||
            (opt.micro_shift_steps > 0 && opt.micro_shift_eps > 0.0);
        const bool use_micro = stats.use_micro;
        const MicroAdjustOptions micro_opt{
            opt.micro_rot_eps,
            opt.micro_rot_steps,
            opt.micro_shift_eps,
            opt.micro_shift_steps,
            9
        };

        GeometryContext geom{base_poly, radius};
        EvalContext eval{geom, opt.final_rigid};
        PostprocessContext post{geom, opt.final_rigid, use_micro, micro_opt};

        for (int n = 1; n <= kNMax; ++n) {
            const bool debug_this_n = (opt.debug_n == n);
            SelectionResult sel = select_best_for_n(eval, subs, n, debug_this_n);
            if (debug_this_n) {
                DebugContext dbg{n, opt.debug_top, opt.inputs};
                print_debug_for_n(dbg, sel.debug_entries, sel.best_k);
            }

            BuiltSolution sol = build_solution(post, std::move(sel.best_poses));
            if (sol.micro_improved) {
                ++stats.micro_improved;
            }
            out_by_n[static_cast<size_t>(n)] = std::move(sol.poses);
            side_by_n[static_cast<size_t>(n)] = sol.side;
        }

        stats.cross_fixes = opt.cross_check_n ? apply_cross_check(base_poly, out_by_n, side_by_n) : 0;
        stats.total_score = write_output(opt.output_path, out_by_n, side_by_n);
        print_summary(opt, stats);

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
