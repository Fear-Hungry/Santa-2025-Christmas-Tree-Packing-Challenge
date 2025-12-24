#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry/geom.hpp"
#include "solvers/post_opt.hpp"
#include "utils/cli_parse.hpp"
#include "utils/submission_io.hpp"

namespace {

void print_usage(const char* argv0) {
    std::cout << "Uso: " << argv0 << " --input <csv> --output <csv> [opcoes]\n"
              << "Opcoes:\n"
              << "  --iters <int>            Iteracoes SA (default 15000)\n"
              << "  --restarts <int>         Restarts SA (default 16)\n"
              << "  --t0 <double>            Temperatura inicial (default 2.5)\n"
              << "  --tm <double>            Temperatura minima (default 5e-7)\n"
              << "  --remove-ratio <double>  Fracao removida no free-area (default 0.5)\n"
              << "  --free-area-min-n <int>  N minimo para free-area (default 10)\n"
              << "  --backprop-passes <int>  Passes de back-prop (default 10)\n"
              << "  --backprop-span <int>    Alcance k+1..k+span (default 5)\n"
              << "  --backprop-max-combos <int>  Max combos aleatorios (default 200)\n"
              << "  --enable-term-scheduler  Ativa scheduler por termo (default off)\n"
              << "  --term-epochs <int>      Epochs do scheduler (default 3)\n"
              << "  --term-tier-a <int>      Top-N em tier A (default 20)\n"
              << "  --term-tier-b <int>      Top-N em tier B (default 120)\n"
              << "  --term-min-n <int>       Ignora n < min_n no ranking (default 1)\n"
              << "  --tier-a-iters-mult <double>    Multiplicador iters tier A (default 1.5)\n"
              << "  --tier-a-restarts-mult <double> Multiplicador restarts tier A (default 1.0)\n"
              << "  --tier-b-iters-mult <double>    Multiplicador iters tier B (default 0.5)\n"
              << "  --tier-b-restarts-mult <double> Multiplicador restarts tier B (default 0.25)\n"
              << "  --tier-c-iters-mult <double>    Multiplicador iters tier C (default 0.0)\n"
              << "  --tier-c-restarts-mult <double> Multiplicador restarts tier C (default 0.0)\n"
              << "  --tier-a-tighten-mult <double>  Multiplicador tighten tier A (default 1.0)\n"
              << "  --tier-b-tighten-mult <double>  Multiplicador tighten tier B (default 0.4)\n"
              << "  --tier-c-tighten-mult <double>  Multiplicador tighten tier C (default 0.1)\n"
              << "  --accept-term-eps <double>  Aceita piora <= eps no termo (default 0.0)\n"
              << "  --enable-guided-reinsert  Ativa reinsercao guiada (default off)\n"
              << "  --reinsert-attempts-tier-a <int> Tentativas reinsercao tier A (default 4000)\n"
              << "  --reinsert-attempts-tier-b <int> Tentativas reinsercao tier B (default 1200)\n"
              << "  --reinsert-attempts-tier-c <int> Tentativas reinsercao tier C (default 200)\n"
              << "  --reinsert-shell-anchors <int>  Anchors shell (default 32)\n"
              << "  --reinsert-core-anchors <int>   Anchors core (default 64)\n"
              << "  --reinsert-jitter-attempts <int> Jitters por anchor (default 256)\n"
              << "  --reinsert-angle-jitter-deg <double>  Jitter de angulo (default 30)\n"
              << "  --reinsert-early-stop-rel <double>  Early-stop por melhora relativa (default 0.002)\n"
              << "  --enable-backprop-explore  Permite backprop exploratorio (default off)\n"
              << "  --backprop-span-tier-a <int> Span backprop tier A (default 10)\n"
              << "  --backprop-span-tier-b <int> Span backprop tier B (default 7)\n"
              << "  --backprop-span-tier-c <int> Span backprop tier C (default 5)\n"
              << "  --backprop-max-combos-tier-a <int> Max combos tier A (default 400)\n"
              << "  --backprop-max-combos-tier-b <int> Max combos tier B (default 250)\n"
              << "  --backprop-max-combos-tier-c <int> Max combos tier C (default 200)\n"
              << "  --threads <int>          Threads OpenMP (0=auto)\n"
              << "  --seed <u64>             Seed base\n"
              << "  --no-free-area           Desativa free-area\n"
              << "  --no-backprop            Desativa back-prop\n"
              << "  --no-squeeze             Desativa squeeze\n"
              << "  --no-compaction          Desativa compaction\n"
              << "  --no-edge-slide          Desativa edge-slide\n"
              << "  --no-local-search        Desativa local search\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input_path = "submission.csv";
        std::string output_path = "submission_post.csv";
        PostOptOptions opt;
        opt.enabled = true;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            }
            if (arg == "--input" || arg == "-i") {
                input_path = require_arg(i, argc, argv, arg);
                continue;
            }
            if (arg == "--output" || arg == "-o") {
                output_path = require_arg(i, argc, argv, arg);
                continue;
            }
            if (arg == "--iters") {
                opt.iters = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--restarts") {
                opt.restarts = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--t0") {
                opt.t0 = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tm") {
                opt.tm = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--remove-ratio") {
                opt.remove_ratio = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--free-area-min-n") {
                opt.free_area_min_n = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--backprop-passes") {
                opt.backprop_passes = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--backprop-span") {
                opt.backprop_span = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--backprop-max-combos") {
                opt.backprop_max_combos = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--enable-term-scheduler") {
                opt.enable_term_scheduler = true;
                continue;
            }
            if (arg == "--term-epochs") {
                opt.term_epochs = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--term-tier-a") {
                opt.term_tier_a = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--term-tier-b") {
                opt.term_tier_b = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--term-min-n") {
                opt.term_min_n = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-a-iters-mult") {
                opt.tier_a_iters_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-a-restarts-mult") {
                opt.tier_a_restarts_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-b-iters-mult") {
                opt.tier_b_iters_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-b-restarts-mult") {
                opt.tier_b_restarts_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-c-iters-mult") {
                opt.tier_c_iters_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-c-restarts-mult") {
                opt.tier_c_restarts_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-a-tighten-mult") {
                opt.tier_a_tighten_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-b-tighten-mult") {
                opt.tier_b_tighten_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--tier-c-tighten-mult") {
                opt.tier_c_tighten_mult = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--accept-term-eps") {
                opt.accept_term_eps = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--enable-guided-reinsert") {
                opt.enable_guided_reinsert = true;
                continue;
            }
            if (arg == "--reinsert-attempts-tier-a") {
                opt.reinsert_attempts_tier_a = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--reinsert-attempts-tier-b") {
                opt.reinsert_attempts_tier_b = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--reinsert-attempts-tier-c") {
                opt.reinsert_attempts_tier_c = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--reinsert-shell-anchors") {
                opt.reinsert_shell_anchors = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--reinsert-core-anchors") {
                opt.reinsert_core_anchors = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--reinsert-jitter-attempts") {
                opt.reinsert_jitter_attempts = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--reinsert-angle-jitter-deg") {
                opt.reinsert_angle_jitter_deg = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--reinsert-early-stop-rel") {
                opt.reinsert_early_stop_rel = parse_double(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--enable-backprop-explore") {
                opt.enable_backprop_explore = true;
                continue;
            }
            if (arg == "--backprop-span-tier-a") {
                opt.backprop_span_tier_a = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--backprop-span-tier-b") {
                opt.backprop_span_tier_b = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--backprop-span-tier-c") {
                opt.backprop_span_tier_c = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--backprop-max-combos-tier-a") {
                opt.backprop_max_combos_tier_a = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--backprop-max-combos-tier-b") {
                opt.backprop_max_combos_tier_b = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--backprop-max-combos-tier-c") {
                opt.backprop_max_combos_tier_c = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--threads") {
                opt.threads = parse_int(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--seed") {
                opt.seed = parse_u64(require_arg(i, argc, argv, arg));
                continue;
            }
            if (arg == "--no-free-area") {
                opt.enable_free_area = false;
                continue;
            }
            if (arg == "--no-backprop") {
                opt.enable_backprop = false;
                continue;
            }
            if (arg == "--no-squeeze") {
                opt.enable_squeeze = false;
                continue;
            }
            if (arg == "--no-compaction") {
                opt.enable_compaction = false;
                continue;
            }
            if (arg == "--no-edge-slide") {
                opt.enable_edge_slide = false;
                continue;
            }
            if (arg == "--no-local-search") {
                opt.enable_local_search = false;
                continue;
            }
            throw std::runtime_error("Argumento desconhecido: " + arg);
        }

        if (opt.term_min_n < 1 || opt.term_min_n > 200) {
            throw std::runtime_error("--term-min-n must be in [1,200].");
        }
        if (opt.tier_a_tighten_mult < 0.0 || opt.tier_b_tighten_mult < 0.0 ||
            opt.tier_c_tighten_mult < 0.0) {
            throw std::runtime_error("tier tighten multipliers must be >= 0.");
        }
        if (opt.reinsert_early_stop_rel < 0.0 || opt.reinsert_early_stop_rel > 1.0) {
            throw std::runtime_error("--reinsert-early-stop-rel must be in [0,1].");
        }

        SubmissionPoses sub = load_submission_poses(input_path, 200);
        std::vector<std::vector<TreePose>> solutions = sub.by_n;
        Polygon base_poly = get_tree_polygon();

        PostOptStats stats;
        post_optimize_submission(base_poly, solutions, opt, &stats);

        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("Erro ao abrir arquivo de saida: " + output_path);
        }
        out << "id,x,y,deg\n";
        for (int n = 1; n <= 200; ++n) {
            if (solutions[static_cast<size_t>(n)].size() != static_cast<size_t>(n)) {
                throw std::runtime_error("Submission incompleta para n=" + std::to_string(n));
            }
            for (int i = 0; i < n; ++i) {
                const TreePose& pose = solutions[static_cast<size_t>(n)][static_cast<size_t>(i)];
                out << fmt_submission_id(n, i) << ","
                    << fmt_submission_value(pose.x) << ","
                    << fmt_submission_value(pose.y) << ","
                    << fmt_submission_value(pose.deg) << "\n";
            }
        }

        std::cout << std::fixed << std::setprecision(9);
        std::cout << "Post-opt complete\n";
        std::cout << "Initial score: " << stats.initial_score << "\n";
        std::cout << "Final score:   " << stats.final_score << "\n";
        std::cout << "Phase1 improved: " << stats.phase1_improved << "\n";
        std::cout << "Backprop improved: " << stats.backprop_improved << "\n";
        std::cout << "Elapsed: " << stats.elapsed_sec << "s\n";
        if (opt.enable_term_scheduler && !stats.term_epochs.empty()) {
            const PostOptTermSummary& ts = stats.term_summary;
            const double drop_a = ts.tier_a_term_before - ts.tier_a_term_after;
            const double drop_b = ts.tier_b_term_before - ts.tier_b_term_after;
            const int k = ts.tier_a_count + ts.tier_b_count;
            std::cout << "Term tiers: A=" << ts.tier_a_count
                      << ", B=" << ts.tier_b_count
                      << " (K=" << k << ")"
                      << ", epochs=" << opt.term_epochs
                      << ", min_n=" << opt.term_min_n << "\n";
            std::cout << "Tier A term: " << ts.tier_a_term_before
                      << " -> " << ts.tier_a_term_after
                      << " (drop=" << drop_a << ")\n";
            std::cout << "Tier B term: " << ts.tier_b_term_before
                      << " -> " << ts.tier_b_term_after
                      << " (drop=" << drop_b << ")\n";
            std::cout << "Tier A n: {";
            for (size_t i = 0; i < ts.tier_a_ns.size(); ++i) {
                if (i) {
                    std::cout << ",";
                }
                std::cout << ts.tier_a_ns[i];
            }
            std::cout << "}\n";
            for (const PostOptTermEpochStats& e : stats.term_epochs) {
                const double edrop_a = e.tier_a_term_before - e.tier_a_term_after;
                const double edrop_b = e.tier_b_term_before - e.tier_b_term_after;
                std::cout << "Epoch " << e.epoch
                          << ": A drop=" << edrop_a
                          << ", B drop=" << edrop_b << "\n";
            }
        }
        std::cout << "Saved: " << output_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Erro no post_opt: " << ex.what() << "\n";
        return 1;
    }
}
