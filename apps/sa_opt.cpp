#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "santa2025/bottom_left.hpp"
#include "santa2025/grid_shake.hpp"
#include "santa2025/lns.hpp"
#include "santa2025/logging.hpp"
#include "santa2025/simulated_annealing.hpp"
#include "santa2025/submission_csv.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    std::string init = "bottom-left";  // bottom-left | grid-shake
    std::vector<std::string> inits{};  // optional multi-start list

    int n = 200;
    std::vector<double> angles{45.0};
    std::vector<double> cycle{};
    std::string mode = "all";
    int cycle_prefix = 0;

    std::string objective = "s";  // s | score
    int nmax_score = 200;
    double target_side = 0.0;
    int target_power = 2;

    std::string delta_mode = "squared_over_s";  // linear | squared | squared_over_s | squared_over_s2
    double delta_scale = 1.0;
    std::string secondary = "none";  // none | perimeter | area | aspect
    double secondary_weight = 0.0;

    std::uint64_t seed = 1;
    int iters = 200000;
    int tries = 8;
    int runs = 1;
    int threads = 1;
    double t0 = 0.25;
    double t1 = 1e-4;
    std::string schedule = "geometric";  // geometric | poly
    double poly_power = 1.0;
    int adaptive_window = 0;
    double target_accept = 0.35;

    double boundary_prob = 0.5;
    double cluster_prob = 0.08;
    int cluster_min = 2;
    int cluster_max = 8;
    double cluster_radius_mult = 4.0;
    double cluster_sigma_mult = 2.5;
    int touch_best_of = 1;

    double touch_prob = 0.25;
    double rot_prob = 0.15;
    double sigma0 = 0.20;
    double sigma1 = 0.01;

    double gap = 1e-6;
    int max_offsets = 512;
    int log_every = 5000;
    double eps = 1e-12;

    bool shrink_wrap = false;
    int shrink_stages = 50;
    int shrink_stage_iters = 5000;
    double shrink_delta = 0.0;
    double shrink_factor = 0.999;

    bool lns = false;
    int lns_stages = 50;
    int lns_stage_attempts = 50;
    double lns_remove_frac = 0.10;
    double lns_remove_frac_max = 0.10;
    double lns_remove_frac_growth = 1.0;
    int lns_remove_frac_growth_every = 1;
    double lns_boundary_prob = 0.7;
    std::string lns_destroy_mode = "mix-random-boundary";  // mix-random-boundary|random|boundary|cluster|gap
    int lns_gap_grid = 48;
    int lns_gap_try_hole_center = 1;
    int lns_slide_iters = 60;
    double lns_shrink_factor = 0.999;
    double lns_shrink_delta = 0.0;
    int lns_repair_sa_iters = 0;
    int lns_repair_sa_best_of = 2;
    double lns_repair_sa_t0 = 0.02;
    double lns_repair_sa_t1 = 1e-4;
    int lns_repair_sa_anchor_samples = 2;

    std::string out_json;
    std::string out_csv;
    int csv_nmax = 200;
    int csv_precision = 17;
};

std::vector<double> parse_angles(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
            continue;
        }
        out.push_back(std::stod(tok));
    }
    return out;
}

std::vector<std::string> parse_csv_strings(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
            continue;
        }
        out.push_back(tok);
    }
    return out;
}

santa2025::LNSDestroyMode parse_lns_destroy_mode(const std::string& s) {
    if (s == "mix-random-boundary") {
        return santa2025::LNSDestroyMode::kMixRandomBoundary;
    }
    if (s == "random") {
        return santa2025::LNSDestroyMode::kRandom;
    }
    if (s == "boundary") {
        return santa2025::LNSDestroyMode::kBoundary;
    }
    if (s == "cluster") {
        return santa2025::LNSDestroyMode::kCluster;
    }
    if (s == "gap") {
        return santa2025::LNSDestroyMode::kGap;
    }
    throw std::runtime_error("invalid --lns-destroy-mode (use mix-random-boundary|random|boundary|cluster|gap)");
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return std::string(argv[++i]);
        };

        if (a == "--init") {
            args.init = need("--init");
        } else if (a == "--inits") {
            args.inits = parse_csv_strings(need("--inits"));
        } else if (a == "--n") {
            args.n = std::stoi(need("--n"));
        } else if (a == "--angles") {
            args.angles = parse_angles(need("--angles"));
        } else if (a == "--cycle") {
            args.cycle = parse_angles(need("--cycle"));
        } else if (a == "--mode") {
            args.mode = need("--mode");
        } else if (a == "--cycle-prefix") {
            args.cycle_prefix = std::stoi(need("--cycle-prefix"));
        } else if (a == "--objective") {
            args.objective = need("--objective");
        } else if (a == "--nmax-score") {
            args.nmax_score = std::stoi(need("--nmax-score"));
        } else if (a == "--target-side") {
            args.target_side = std::stod(need("--target-side"));
        } else if (a == "--target-power") {
            args.target_power = std::stoi(need("--target-power"));
        } else if (a == "--delta-mode") {
            args.delta_mode = need("--delta-mode");
        } else if (a == "--delta-scale") {
            args.delta_scale = std::stod(need("--delta-scale"));
        } else if (a == "--secondary") {
            args.secondary = need("--secondary");
        } else if (a == "--secondary-weight") {
            args.secondary_weight = std::stod(need("--secondary-weight"));
        } else if (a == "--seed") {
            args.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
        } else if (a == "--iters") {
            args.iters = std::stoi(need("--iters"));
        } else if (a == "--tries") {
            args.tries = std::stoi(need("--tries"));
        } else if (a == "--runs") {
            args.runs = std::stoi(need("--runs"));
        } else if (a == "--threads") {
            args.threads = std::stoi(need("--threads"));
        } else if (a == "--t0") {
            args.t0 = std::stod(need("--t0"));
        } else if (a == "--t1") {
            args.t1 = std::stod(need("--t1"));
        } else if (a == "--schedule") {
            args.schedule = need("--schedule");
        } else if (a == "--poly-power") {
            args.poly_power = std::stod(need("--poly-power"));
        } else if (a == "--adaptive-window") {
            args.adaptive_window = std::stoi(need("--adaptive-window"));
        } else if (a == "--target-accept") {
            args.target_accept = std::stod(need("--target-accept"));
        } else if (a == "--boundary-prob") {
            args.boundary_prob = std::stod(need("--boundary-prob"));
        } else if (a == "--cluster-prob") {
            args.cluster_prob = std::stod(need("--cluster-prob"));
        } else if (a == "--cluster-min") {
            args.cluster_min = std::stoi(need("--cluster-min"));
        } else if (a == "--cluster-max") {
            args.cluster_max = std::stoi(need("--cluster-max"));
        } else if (a == "--cluster-radius-mult") {
            args.cluster_radius_mult = std::stod(need("--cluster-radius-mult"));
        } else if (a == "--cluster-sigma-mult") {
            args.cluster_sigma_mult = std::stod(need("--cluster-sigma-mult"));
        } else if (a == "--touch-best-of") {
            args.touch_best_of = std::stoi(need("--touch-best-of"));
        } else if (a == "--touch-prob") {
            args.touch_prob = std::stod(need("--touch-prob"));
        } else if (a == "--rot-prob") {
            args.rot_prob = std::stod(need("--rot-prob"));
        } else if (a == "--sigma0") {
            args.sigma0 = std::stod(need("--sigma0"));
        } else if (a == "--sigma1") {
            args.sigma1 = std::stod(need("--sigma1"));
        } else if (a == "--gap") {
            args.gap = std::stod(need("--gap"));
        } else if (a == "--max-offsets") {
            args.max_offsets = std::stoi(need("--max-offsets"));
        } else if (a == "--log-every") {
            args.log_every = std::stoi(need("--log-every"));
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "--shrink-wrap") {
            args.shrink_wrap = true;
        } else if (a == "--shrink-stages") {
            args.shrink_stages = std::stoi(need("--shrink-stages"));
        } else if (a == "--shrink-stage-iters") {
            args.shrink_stage_iters = std::stoi(need("--shrink-stage-iters"));
        } else if (a == "--shrink-delta") {
            args.shrink_delta = std::stod(need("--shrink-delta"));
        } else if (a == "--shrink-factor") {
            args.shrink_factor = std::stod(need("--shrink-factor"));
        } else if (a == "--lns") {
            args.lns = true;
        } else if (a == "--lns-stages") {
            args.lns_stages = std::stoi(need("--lns-stages"));
        } else if (a == "--lns-stage-attempts") {
            args.lns_stage_attempts = std::stoi(need("--lns-stage-attempts"));
        } else if (a == "--lns-remove-frac") {
            args.lns_remove_frac = std::stod(need("--lns-remove-frac"));
        } else if (a == "--lns-remove-frac-max") {
            args.lns_remove_frac_max = std::stod(need("--lns-remove-frac-max"));
        } else if (a == "--lns-remove-frac-growth") {
            args.lns_remove_frac_growth = std::stod(need("--lns-remove-frac-growth"));
        } else if (a == "--lns-remove-frac-growth-every") {
            args.lns_remove_frac_growth_every = std::stoi(need("--lns-remove-frac-growth-every"));
        } else if (a == "--lns-boundary-prob") {
            args.lns_boundary_prob = std::stod(need("--lns-boundary-prob"));
        } else if (a == "--lns-destroy-mode") {
            args.lns_destroy_mode = need("--lns-destroy-mode");
        } else if (a == "--lns-gap-grid") {
            args.lns_gap_grid = std::stoi(need("--lns-gap-grid"));
        } else if (a == "--lns-gap-try-hole-center") {
            args.lns_gap_try_hole_center = std::stoi(need("--lns-gap-try-hole-center"));
        } else if (a == "--lns-slide-iters") {
            args.lns_slide_iters = std::stoi(need("--lns-slide-iters"));
        } else if (a == "--lns-shrink-factor") {
            args.lns_shrink_factor = std::stod(need("--lns-shrink-factor"));
        } else if (a == "--lns-shrink-delta") {
            args.lns_shrink_delta = std::stod(need("--lns-shrink-delta"));
        } else if (a == "--lns-repair-sa-iters") {
            args.lns_repair_sa_iters = std::stoi(need("--lns-repair-sa-iters"));
        } else if (a == "--lns-repair-sa-best-of") {
            args.lns_repair_sa_best_of = std::stoi(need("--lns-repair-sa-best-of"));
        } else if (a == "--lns-repair-sa-t0") {
            args.lns_repair_sa_t0 = std::stod(need("--lns-repair-sa-t0"));
        } else if (a == "--lns-repair-sa-t1") {
            args.lns_repair_sa_t1 = std::stod(need("--lns-repair-sa-t1"));
        } else if (a == "--lns-repair-sa-anchor-samples") {
            args.lns_repair_sa_anchor_samples = std::stoi(need("--lns-repair-sa-anchor-samples"));
        } else if (a == "--out-json") {
            args.out_json = need("--out-json");
        } else if (a == "--out-csv") {
            args.out_csv = need("--out-csv");
        } else if (a == "--csv-nmax") {
            args.csv_nmax = std::stoi(need("--csv-nmax"));
        } else if (a == "--csv-precision") {
            args.csv_precision = std::stoi(need("--csv-precision"));
        } else if (a == "-h" || a == "--help") {
            std::cout
                << "Usage: sa_opt [--init bottom-left|grid-shake] [--inits bottom-left,grid-shake] [--n N]\n"
                << "             [--angles a,b,c]\n"
                << "             [--cycle a,b,c] [--mode all|cycle|cycle-then-all] [--cycle-prefix N]\n"
                << "             [--objective s|score|target] [--nmax-score N] [--target-side S] [--target-power 1|2]\n"
                << "             [--delta-mode linear|squared|squared_over_s|squared_over_s2] [--delta-scale x]\n"
                << "             [--secondary none|perimeter|area|aspect] [--secondary-weight w]\n"
                << "             [--seed S] [--iters N] [--tries K] [--runs R] [--threads T] [--t0 T] [--t1 T]\n"
                << "             [--schedule geometric|poly] [--poly-power p]\n"
                << "             [--adaptive-window N] [--target-accept r]\n"
                << "             [--boundary-prob p] [--cluster-prob p] [--cluster-min k] [--cluster-max k]\n"
                << "             [--cluster-radius-mult r] [--cluster-sigma-mult r] [--touch-best-of k]\n"
                << "             [--touch-prob p] [--rot-prob p] [--sigma0 s] [--sigma1 s]\n"
                << "             [--gap g] [--max-offsets N] [--log-every N] [--eps e]\n"
                << "             [--shrink-wrap] [--shrink-stages N] [--shrink-stage-iters N] [--shrink-delta d]\n"
                << "             [--shrink-factor f]\n"
                << "             [--lns] [--lns-stages N] [--lns-stage-attempts N]\n"
                << "             [--lns-remove-frac f] [--lns-remove-frac-max f] [--lns-remove-frac-growth g]\n"
                << "             [--lns-remove-frac-growth-every k] [--lns-boundary-prob p]\n"
                << "             [--lns-destroy-mode mix-random-boundary|random|boundary|cluster|gap]\n"
                << "             [--lns-gap-grid N] [--lns-gap-try-hole-center 0|1]\n"
                << "             [--lns-slide-iters N]\n"
                << "             [--lns-shrink-factor f] [--lns-shrink-delta d]\n"
                << "             [--lns-repair-sa-iters N] [--lns-repair-sa-best-of k]\n"
                << "             [--lns-repair-sa-t0 T] [--lns-repair-sa-t1 T]\n"
                << "             [--lns-repair-sa-anchor-samples k]\n"
                << "             [--out-json path] [--out-csv path] [--csv-nmax N] [--csv-precision P]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + a);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);

        if (args.n <= 0 || args.n > 200) {
            throw std::runtime_error("--n must be in [1,200]");
        }
        if (args.runs <= 0) {
            throw std::runtime_error("--runs must be > 0");
        }
        if (args.csv_nmax <= 0 || args.csv_nmax > 200) {
            throw std::runtime_error("--csv-nmax must be in [1,200]");
        }
        if (args.csv_precision < 0 || args.csv_precision > 17) {
            throw std::runtime_error("--csv-precision must be in [0,17]");
        }

        santa2025::OrientationMode mode = santa2025::OrientationMode::kTryAll;
        if (args.mode == "all") {
            mode = santa2025::OrientationMode::kTryAll;
        } else if (args.mode == "cycle") {
            mode = santa2025::OrientationMode::kCycle;
        } else if (args.mode == "cycle-then-all") {
            mode = santa2025::OrientationMode::kCycleThenAll;
        } else {
            throw std::runtime_error("invalid --mode (use all|cycle|cycle-then-all)");
        }

        const santa2025::Polygon poly = santa2025::tree_polygon();

        auto build_initial = [&](const std::string& init_method) -> std::vector<santa2025::Pose> {
            if (init_method == "bottom-left") {
                santa2025::BottomLeftOptions opt;
                opt.n = args.n;
                opt.angles_deg = args.angles;
                opt.cycle_deg = args.cycle;
                opt.orientation_mode = mode;
                opt.cycle_prefix = args.cycle_prefix;
                opt.gap = args.gap;
                opt.max_offsets_per_delta = args.max_offsets;
                return santa2025::bottom_left_pack(poly, opt, args.eps);
            }
            if (init_method == "grid-shake") {
                santa2025::GridShakeOptions opt;
                opt.n = args.n;
                opt.angles_deg = args.angles;
                opt.cycle_deg = args.cycle;
                opt.orientation_mode = mode;
                opt.cycle_prefix = args.cycle_prefix;
                opt.gap = args.gap;
                return santa2025::grid_shake_pack(poly, opt, args.eps);
            }
            throw std::runtime_error("invalid init method (use bottom-left|grid-shake): " + init_method);
        };

        std::vector<std::string> init_methods;
        if (args.inits.empty()) {
            init_methods = {args.init};
        } else {
            init_methods = args.inits;
        }
        if (init_methods.empty()) {
            throw std::runtime_error("empty init method list");
        }

        // Precompute initial packings for each init method used.
        std::unordered_map<std::string, std::vector<santa2025::Pose>> initial_cache;
        if (args.log_every > 0) {
            std::lock_guard<std::mutex> lk(santa2025::log_mutex());
            std::cerr << "[sa_opt] building initial packings for " << init_methods.size() << " init(s)...\n";
        }
        for (const auto& m : init_methods) {
            if (initial_cache.find(m) != initial_cache.end()) {
                continue;
            }
            initial_cache.emplace(m, build_initial(m));
        }

        santa2025::SAOptions sa;
        sa.n = args.n;
        sa.angles_deg = args.angles;
        sa.cycle_deg = args.cycle;
        sa.orientation_mode = mode;
        sa.cycle_prefix = args.cycle_prefix;
        if (args.objective == "s") {
            sa.objective = santa2025::SAObjective::kS200;
        } else if (args.objective == "score") {
            sa.objective = santa2025::SAObjective::kPrefixScore;
        } else if (args.objective == "target") {
            sa.objective = santa2025::SAObjective::kTargetSide;
            sa.target_side = args.target_side;
            sa.target_power = args.target_power;
        } else {
            throw std::runtime_error("invalid --objective (use s|score|target)");
        }
        sa.nmax_score = args.nmax_score;
        if (args.delta_mode == "linear") {
            sa.delta_mode = santa2025::SADeltaMode::kLinear;
        } else if (args.delta_mode == "squared") {
            sa.delta_mode = santa2025::SADeltaMode::kSquared;
        } else if (args.delta_mode == "squared_over_s") {
            sa.delta_mode = santa2025::SADeltaMode::kSquaredOverS;
        } else if (args.delta_mode == "squared_over_s2") {
            sa.delta_mode = santa2025::SADeltaMode::kSquaredOverS2;
        } else {
            throw std::runtime_error("invalid --delta-mode");
        }
        sa.delta_scale = args.delta_scale;
        if (args.secondary == "none") {
            sa.secondary = santa2025::SASecondary::kNone;
        } else if (args.secondary == "perimeter") {
            sa.secondary = santa2025::SASecondary::kPerimeter;
        } else if (args.secondary == "area") {
            sa.secondary = santa2025::SASecondary::kArea;
        } else if (args.secondary == "aspect") {
            sa.secondary = santa2025::SASecondary::kAspect;
        } else {
            throw std::runtime_error("invalid --secondary");
        }
        sa.secondary_weight = args.secondary_weight;
        sa.seed = args.seed;
        sa.iters = args.iters;
        sa.tries_per_iter = args.tries;
        sa.t0 = args.t0;
        sa.t1 = args.t1;
        if (args.schedule == "geometric") {
            sa.schedule = santa2025::SASchedule::kGeometric;
        } else if (args.schedule == "poly" || args.schedule == "polynomial") {
            sa.schedule = santa2025::SASchedule::kPolynomial;
        } else {
            throw std::runtime_error("invalid --schedule (use geometric|poly)");
        }
        sa.poly_power = args.poly_power;
        sa.adaptive_window = args.adaptive_window;
        sa.target_accept = args.target_accept;
        sa.boundary_prob = args.boundary_prob;
        sa.cluster_prob = args.cluster_prob;
        sa.cluster_min = args.cluster_min;
        sa.cluster_max = args.cluster_max;
        sa.cluster_radius_mult = args.cluster_radius_mult;
        sa.cluster_sigma_mult = args.cluster_sigma_mult;
        sa.touch_best_of = args.touch_best_of;
        sa.touch_prob = args.touch_prob;
        sa.rot_prob = args.rot_prob;
        sa.trans_sigma0 = args.sigma0;
        sa.trans_sigma1 = args.sigma1;
        sa.gap = args.gap;
        sa.max_offsets_per_delta = args.max_offsets;
        sa.log_every = args.log_every;

        if (args.shrink_wrap) {
            if (args.objective != "s") {
                throw std::runtime_error("--shrink-wrap currently supports only --objective s");
            }
            if (args.shrink_stages <= 0) {
                throw std::runtime_error("--shrink-stages must be > 0");
            }
            if (args.shrink_stage_iters <= 0) {
                throw std::runtime_error("--shrink-stage-iters must be > 0");
            }
            if (!(args.shrink_factor > 0.0 && args.shrink_factor <= 1.0)) {
                throw std::runtime_error("--shrink-factor must be in (0,1]");
            }
            if (!(args.shrink_delta >= 0.0)) {
                throw std::runtime_error("--shrink-delta must be >= 0");
            }
        }
        if (args.lns) {
            if (args.lns_stages < 0) {
                throw std::runtime_error("--lns-stages must be >= 0");
            }
            if (args.lns_stage_attempts <= 0) {
                throw std::runtime_error("--lns-stage-attempts must be > 0");
            }
            if (!(args.lns_remove_frac > 0.0 && args.lns_remove_frac <= 1.0)) {
                throw std::runtime_error("--lns-remove-frac must be in (0,1]");
            }
            if (!(args.lns_remove_frac_max > 0.0 && args.lns_remove_frac_max <= 1.0)) {
                throw std::runtime_error("--lns-remove-frac-max must be in (0,1]");
            }
            if (!(args.lns_remove_frac_max + 1e-12 >= args.lns_remove_frac)) {
                throw std::runtime_error("--lns-remove-frac-max must be >= --lns-remove-frac");
            }
            if (!(args.lns_remove_frac_growth >= 1.0)) {
                throw std::runtime_error("--lns-remove-frac-growth must be >= 1");
            }
            if (args.lns_remove_frac_growth_every <= 0) {
                throw std::runtime_error("--lns-remove-frac-growth-every must be > 0");
            }
            if (!(args.lns_boundary_prob >= 0.0 && args.lns_boundary_prob <= 1.0)) {
                throw std::runtime_error("--lns-boundary-prob must be in [0,1]");
            }
            (void)parse_lns_destroy_mode(args.lns_destroy_mode);
            if (args.lns_gap_grid < 4) {
                throw std::runtime_error("--lns-gap-grid must be >= 4");
            }
            if (!(args.lns_gap_try_hole_center == 0 || args.lns_gap_try_hole_center == 1)) {
                throw std::runtime_error("--lns-gap-try-hole-center must be 0 or 1");
            }
            if (args.lns_slide_iters <= 0) {
                throw std::runtime_error("--lns-slide-iters must be > 0");
            }
            if (!(args.lns_shrink_factor > 0.0 && args.lns_shrink_factor <= 1.0)) {
                throw std::runtime_error("--lns-shrink-factor must be in (0,1]");
            }
            if (!(args.lns_shrink_delta >= 0.0)) {
                throw std::runtime_error("--lns-shrink-delta must be >= 0");
            }
            if (args.lns_repair_sa_iters < 0) {
                throw std::runtime_error("--lns-repair-sa-iters must be >= 0");
            }
            if (args.lns_repair_sa_best_of <= 0) {
                throw std::runtime_error("--lns-repair-sa-best-of must be >= 1");
            }
            if (!(args.lns_repair_sa_t0 > 0.0) || !(args.lns_repair_sa_t1 > 0.0)) {
                throw std::runtime_error("--lns-repair-sa-t0/--lns-repair-sa-t1 must be > 0");
            }
            if (args.lns_repair_sa_anchor_samples <= 0) {
                throw std::runtime_error("--lns-repair-sa-anchor-samples must be >= 1");
            }
            if (args.objective == "target") {
                throw std::runtime_error("--lns is not supported with --objective target");
            }
        }

        struct RunOut {
            int run = 0;
            std::string init_method;
            std::uint64_t seed = 0;
            int shrink_stages_done = 0;
            double shrink_last_target = 0.0;
            santa2025::SAResult res;
            santa2025::LNSResult lns_res;
            std::string error;
        };

        auto run_one = [&](int run_id) -> RunOut {
            RunOut out;
            out.run = run_id;
            out.init_method = init_methods[static_cast<size_t>(run_id) % init_methods.size()];

            constexpr std::uint64_t kSeedStride = 1'000'003ULL;
            out.seed = args.seed + static_cast<std::uint64_t>(run_id) * kSeedStride;

            const auto& initial = initial_cache.at(out.init_method);

            santa2025::SAOptions sa_run = sa;
            sa_run.seed = out.seed;
            sa_run.log_every = args.log_every;
            sa_run.log_prefix = "[run " + std::to_string(run_id) + " sa]";

            const auto t_start = std::chrono::steady_clock::now();
            if (args.log_every > 0) {
                std::lock_guard<std::mutex> lk(santa2025::log_mutex());
                std::cerr << "[run " << run_id << "] start init=" << out.init_method << " seed=" << out.seed
                          << " n=" << args.n << " objective=" << args.objective << " iters=" << args.iters
                          << " tries=" << args.tries << "\n";
            }

            try {
                if (!args.shrink_wrap) {
                    out.res = santa2025::simulated_annealing(poly, initial, sa_run, args.eps);
                } else {
                    std::vector<santa2025::Pose> cur = initial;
                    const double init_s200 = santa2025::packing_s200(poly, cur);
                    const double init_score = santa2025::packing_prefix_score(poly, cur, 200);

                    int attempted = 0;
                    int feasible = 0;
                    int accepted = 0;

                    double cur_s = init_s200;
                    for (int stage = 0; stage < args.shrink_stages; ++stage) {
                        const double target = cur_s * args.shrink_factor - args.shrink_delta;
                        if (!(target > 0.0)) {
                            break;
                        }
                        out.shrink_last_target = target;

                        santa2025::SAOptions st = sa_run;
                        st.objective = santa2025::SAObjective::kTargetSide;
                        st.target_side = target;
                        st.target_power = args.target_power;
                        st.iters = args.shrink_stage_iters;
                        st.seed = out.seed + static_cast<std::uint64_t>(stage);
                        st.log_every = args.log_every;
                        st.log_prefix = "[run " + std::to_string(run_id) + " shrink " + std::to_string(stage) + " sa]";

                        const auto stage_res = santa2025::simulated_annealing(poly, cur, st, args.eps);
                        attempted += stage_res.attempted;
                        feasible += stage_res.feasible;
                        accepted += stage_res.accepted;

                        if (stage_res.best_s200 <= target + 1e-12) {
                            cur = stage_res.best_poses;
                            cur_s = stage_res.best_s200;
                            out.shrink_stages_done++;
                            continue;
                        }
                        break;
                    }

                    santa2025::SAOptions final_sa = sa_run;
                    final_sa.objective = santa2025::SAObjective::kS200;
                    final_sa.seed = out.seed + static_cast<std::uint64_t>(args.shrink_stages);
                    final_sa.log_every = args.log_every;
                    final_sa.log_prefix = "[run " + std::to_string(run_id) + " final sa]";

                    out.res = santa2025::simulated_annealing(poly, cur, final_sa, args.eps);
                    out.res.attempted += attempted;
                    out.res.feasible += feasible;
                    out.res.accepted += accepted;
                    out.res.init_s200 = init_s200;
                    out.res.init_prefix_score = init_score;
                    out.res.init_cost = init_s200;
                }

                // Optional LNS post-pass (works on a feasible packing).
                if (args.lns) {
                    santa2025::LNSOptions lns;
                    lns.n = args.n;
                    lns.angles_deg = args.angles;
                    lns.cycle_deg = args.cycle;
                    lns.orientation_mode = mode;
                    lns.cycle_prefix = args.cycle_prefix;
                    lns.stages = args.lns_stages;
                    lns.stage_attempts = args.lns_stage_attempts;
                    lns.shrink_factor = args.lns_shrink_factor;
                    lns.shrink_delta = args.lns_shrink_delta;
                    lns.remove_frac = args.lns_remove_frac;
                    lns.remove_frac_max = args.lns_remove_frac_max;
                    lns.remove_frac_growth = args.lns_remove_frac_growth;
                    lns.remove_frac_growth_every = args.lns_remove_frac_growth_every;
                    lns.boundary_prob = args.lns_boundary_prob;
                    lns.destroy_mode = parse_lns_destroy_mode(args.lns_destroy_mode);
                    lns.gap_grid = args.lns_gap_grid;
                    lns.gap_try_hole_center = (args.lns_gap_try_hole_center != 0);
                    lns.slide_iters = args.lns_slide_iters;
                    lns.gap = args.gap;
                    lns.repair_sa_iters = args.lns_repair_sa_iters;
                    lns.repair_sa_best_of = args.lns_repair_sa_best_of;
                    lns.repair_sa_t0 = args.lns_repair_sa_t0;
                    lns.repair_sa_t1 = args.lns_repair_sa_t1;
                    lns.repair_sa_anchor_samples = args.lns_repair_sa_anchor_samples;
                    lns.max_offsets_per_delta = args.max_offsets;
                    lns.seed = out.seed + 777'777ULL;
                    lns.log_every = args.log_every;
                    lns.log_prefix = "[run " + std::to_string(run_id) + " lns]";

                    out.lns_res = santa2025::lns_shrink_wrap(poly, out.res.best_poses, lns, args.eps);
                    out.res.best_poses = out.lns_res.best_poses;
                    out.res.best_s200 = santa2025::packing_s200(poly, out.res.best_poses);
                    out.res.best_prefix_score =
                        santa2025::packing_prefix_score(poly, out.res.best_poses, sa_run.nmax_score);
                    out.res.best_cost = (args.objective == "score") ? out.res.best_prefix_score : out.res.best_s200;
                }

                if (args.log_every > 0) {
                    const auto t_end = std::chrono::steady_clock::now();
                    const double secs =
                        std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
                    std::lock_guard<std::mutex> lk(santa2025::log_mutex());
                    std::cerr << "[run " << run_id << "] done best_cost=" << out.res.best_cost
                              << " best_s200=" << out.res.best_s200 << " best_score200=" << out.res.best_prefix_score
                              << " attempted=" << out.res.attempted << " feasible=" << out.res.feasible
                              << " accepted=" << out.res.accepted << " secs=" << secs << "\n";
                }

                return out;
            } catch (const std::exception& e) {
                out.error = e.what();
                return out;
            }
        };

        std::vector<RunOut> runs(static_cast<size_t>(args.runs));
        if (args.runs == 1) {
            runs[0] = run_one(0);
        } else {
            int threads = args.threads;
            if (threads <= 0) {
                threads = static_cast<int>(std::thread::hardware_concurrency());
                if (threads <= 0) {
                    threads = 1;
                }
            }
            threads = std::max(1, std::min(threads, args.runs));

            std::atomic<int> next{0};

            auto worker = [&]() {
                for (;;) {
                    const int id = next.fetch_add(1);
                    if (id >= args.runs) {
                        return;
                    }
                    runs[static_cast<size_t>(id)] = run_one(id);
                }
            };

            std::vector<std::thread> pool;
            pool.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t) {
                pool.emplace_back(worker);
            }
            for (auto& th : pool) {
                th.join();
            }
        }

        for (const auto& r : runs) {
            if (!r.error.empty()) {
                throw std::runtime_error("run " + std::to_string(r.run) + " failed: " + r.error);
            }
        }

        auto better = [&](const RunOut& a, const RunOut& b) {
            if (a.res.best_cost != b.res.best_cost) {
                return a.res.best_cost < b.res.best_cost;
            }
            if (a.res.best_s200 != b.res.best_s200) {
                return a.res.best_s200 < b.res.best_s200;
            }
            if (a.res.best_prefix_score != b.res.best_prefix_score) {
                return a.res.best_prefix_score < b.res.best_prefix_score;
            }
            return a.run < b.run;
        };

        int best_run = 0;
        for (int i = 1; i < args.runs; ++i) {
            if (better(runs[static_cast<size_t>(i)], runs[static_cast<size_t>(best_run)])) {
                best_run = i;
            }
        }

        const auto& best = runs[static_cast<size_t>(best_run)];
        const santa2025::SAResult& res = best.res;

        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\n";
        out << "  \"multi_start\": {\"runs\": " << args.runs << ", \"threads\": " << args.threads
            << ", \"best_run\": " << best_run << "},\n";
        out << "  \"shrink_wrap\": {\"enabled\": " << (args.shrink_wrap ? "true" : "false") << ", \"stages_done\": "
            << best.shrink_stages_done << ", \"last_target\": " << best.shrink_last_target << "},\n";
        out << "  \"lns\": {\"enabled\": " << (args.lns ? "true" : "false") << ", \"stages_done\": "
            << best.lns_res.stages_done << ", \"attempted\": " << best.lns_res.attempted
            << ", \"succeeded\": " << best.lns_res.succeeded << ", \"last_target\": " << best.lns_res.last_target
            << "},\n";
        out << "  \"init\": {\"cost\": " << res.init_cost << ", \"s200\": " << res.init_s200
            << ", \"score\": " << res.init_prefix_score << "},\n";
        out << "  \"best\": {\"cost\": " << res.best_cost << ", \"s200\": " << res.best_s200
            << ", \"score\": " << res.best_prefix_score << "},\n";
        out << "  \"counters\": {\"attempted\": " << res.attempted << ", \"feasible\": " << res.feasible
            << ", \"accepted\": " << res.accepted << "},\n";
        out << "  \"runs\": [\n";
        for (size_t i = 0; i < runs.size(); ++i) {
            const auto& r = runs[i];
            out << "    {\"run\": " << r.run << ", \"init\": \"" << r.init_method << "\", \"seed\": " << r.seed
                << ", \"best\": {\"cost\": " << r.res.best_cost << ", \"s200\": " << r.res.best_s200
                << ", \"score\": " << r.res.best_prefix_score << "}, \"lns\": {\"stages_done\": " << r.lns_res.stages_done
                << ", \"attempted\": " << r.lns_res.attempted << ", \"succeeded\": " << r.lns_res.succeeded
                << "}, \"counters\": {\"attempted\": " << r.res.attempted << ", \"feasible\": " << r.res.feasible
                << ", \"accepted\": " << r.res.accepted << "}}";
            if (i + 1 != runs.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ],\n";
        out << "  \"poses\": [\n";
        for (size_t i = 0; i < res.best_poses.size(); ++i) {
            const auto& p = res.best_poses[i];
            out << "    {\"i\": " << i << ", \"x\": " << p.x << ", \"y\": " << p.y << ", \"deg\": " << p.deg << "}";
            if (i + 1 != res.best_poses.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";

        const std::string payload = out.str();
        std::cout << payload;

        if (!args.out_json.empty()) {
            std::ofstream f(args.out_json);
            if (!f) {
                throw std::runtime_error("failed to open --out-json file: " + args.out_json);
            }
            f << payload;
        }

        if (!args.out_csv.empty()) {
            if (args.csv_nmax > static_cast<int>(res.best_poses.size())) {
                throw std::runtime_error("--csv-nmax must be <= --n");
            }
            std::ofstream f(args.out_csv);
            if (!f) {
                throw std::runtime_error("failed to open --out-csv file: " + args.out_csv);
            }
            santa2025::write_prefix_submission_csv(f, res.best_poses, args.csv_nmax, args.csv_precision);
            if (args.log_every > 0) {
                std::lock_guard<std::mutex> lk(santa2025::log_mutex());
                std::cerr << "[sa_opt] wrote submission csv: " << args.out_csv << " (nmax=" << args.csv_nmax
                          << ", precision=" << args.csv_precision << ")\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
