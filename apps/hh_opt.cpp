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
#include "santa2025/hyper_heuristic.hpp"
#include "santa2025/logging.hpp"
#include "santa2025/submission_csv.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    std::string init = "bottom-left";  // bottom-left | grid-shake
    std::vector<std::string> inits{};

    int n = 200;
    std::vector<double> angles{45.0};
    std::vector<double> cycle{};
    std::string mode = "all";
    int cycle_prefix = 0;

    std::string objective = "s";  // s | score
    int nmax_score = 200;

    std::uint64_t seed = 1;
    int runs = 1;
    int threads = 1;

    int hh_iters = 200;
    int lahc_length = 0;
    double ucb_c = 0.25;

    // Operator base budgets.
    int sa_block_iters = 5000;
    int sa_tries = 8;
    double sa_t0 = 0.25;
    double sa_t1 = 1e-4;
    double sa_sigma0 = 0.20;
    double sa_sigma1 = 0.01;

    int lns_stages = 3;
    int lns_stage_attempts = 20;
    double lns_remove_frac = 0.12;
    double lns_boundary_prob = 0.7;
    double lns_shrink_factor = 0.999;
    double lns_shrink_delta = 0.0;
    int lns_slide_iters = 60;

    double gap = 1e-6;
    int max_offsets = 512;
    int log_every = 0;
    double eps = 1e-12;

    std::string out_json;
    std::string out_csv;
    int csv_nmax = 0;        // if 0, uses n
    int csv_precision = 17;  // keep high to avoid rounding overlaps
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
        } else if (a == "--seed") {
            args.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
        } else if (a == "--runs") {
            args.runs = std::stoi(need("--runs"));
        } else if (a == "--threads") {
            args.threads = std::stoi(need("--threads"));
        } else if (a == "--hh-iters") {
            args.hh_iters = std::stoi(need("--hh-iters"));
        } else if (a == "--lahc-length") {
            args.lahc_length = std::stoi(need("--lahc-length"));
        } else if (a == "--ucb-c") {
            args.ucb_c = std::stod(need("--ucb-c"));
        } else if (a == "--sa-iters") {
            args.sa_block_iters = std::stoi(need("--sa-iters"));
        } else if (a == "--sa-tries") {
            args.sa_tries = std::stoi(need("--sa-tries"));
        } else if (a == "--sa-t0") {
            args.sa_t0 = std::stod(need("--sa-t0"));
        } else if (a == "--sa-t1") {
            args.sa_t1 = std::stod(need("--sa-t1"));
        } else if (a == "--sa-sigma0") {
            args.sa_sigma0 = std::stod(need("--sa-sigma0"));
        } else if (a == "--sa-sigma1") {
            args.sa_sigma1 = std::stod(need("--sa-sigma1"));
        } else if (a == "--lns-stages") {
            args.lns_stages = std::stoi(need("--lns-stages"));
        } else if (a == "--lns-stage-attempts") {
            args.lns_stage_attempts = std::stoi(need("--lns-stage-attempts"));
        } else if (a == "--lns-remove-frac") {
            args.lns_remove_frac = std::stod(need("--lns-remove-frac"));
        } else if (a == "--lns-boundary-prob") {
            args.lns_boundary_prob = std::stod(need("--lns-boundary-prob"));
        } else if (a == "--lns-shrink-factor") {
            args.lns_shrink_factor = std::stod(need("--lns-shrink-factor"));
        } else if (a == "--lns-shrink-delta") {
            args.lns_shrink_delta = std::stod(need("--lns-shrink-delta"));
        } else if (a == "--lns-slide-iters") {
            args.lns_slide_iters = std::stoi(need("--lns-slide-iters"));
        } else if (a == "--gap") {
            args.gap = std::stod(need("--gap"));
        } else if (a == "--max-offsets") {
            args.max_offsets = std::stoi(need("--max-offsets"));
        } else if (a == "--log-every") {
            args.log_every = std::stoi(need("--log-every"));
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "--out-json") {
            args.out_json = need("--out-json");
        } else if (a == "--out-csv") {
            args.out_csv = need("--out-csv");
        } else if (a == "--csv-nmax") {
            args.csv_nmax = std::stoi(need("--csv-nmax"));
        } else if (a == "--csv-precision") {
            args.csv_precision = std::stoi(need("--csv-precision"));
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: hh_opt [--init bottom-left|grid-shake] [--inits ...] [--n N] [--angles a,b,c]\n"
                      << "              [--cycle a,b,c] [--mode all|cycle|cycle-then-all] [--cycle-prefix N]\n"
                      << "              [--objective s|score] [--nmax-score N]\n"
                      << "              [--seed S] [--runs R] [--threads T]\n"
                      << "              [--hh-iters N] [--lahc-length L] [--ucb-c c]\n"
                      << "              [--sa-iters N] [--sa-tries K] [--sa-t0 T] [--sa-t1 T]\n"
                      << "              [--sa-sigma0 s] [--sa-sigma1 s]\n"
                      << "              [--lns-stages N] [--lns-stage-attempts N] [--lns-remove-frac f]\n"
                      << "              [--lns-boundary-prob p] [--lns-shrink-factor f] [--lns-shrink-delta d]\n"
                      << "              [--lns-slide-iters N]\n"
                      << "              [--gap g] [--max-offsets N] [--log-every N] [--eps e]\n"
                      << "              [--out-json path] [--out-csv path] [--csv-nmax N] [--csv-precision P]\n";
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
        if (args.hh_iters <= 0) {
            throw std::runtime_error("--hh-iters must be > 0");
        }
        if (!(args.ucb_c >= 0.0)) {
            throw std::runtime_error("--ucb-c must be >= 0");
        }
        if (args.sa_block_iters <= 0 || args.sa_tries <= 0) {
            throw std::runtime_error("--sa-iters/--sa-tries must be > 0");
        }
        if (!(args.lns_remove_frac > 0.0 && args.lns_remove_frac <= 1.0)) {
            throw std::runtime_error("--lns-remove-frac must be in (0,1]");
        }
        if (!(args.lns_boundary_prob >= 0.0 && args.lns_boundary_prob <= 1.0)) {
            throw std::runtime_error("--lns-boundary-prob must be in [0,1]");
        }
        if (!(args.lns_shrink_factor > 0.0 && args.lns_shrink_factor <= 1.0)) {
            throw std::runtime_error("--lns-shrink-factor must be in (0,1]");
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

        std::vector<std::string> init_methods = args.inits.empty() ? std::vector<std::string>{args.init} : args.inits;
        std::unordered_map<std::string, std::vector<santa2025::Pose>> initial_cache;
        for (const auto& m : init_methods) {
            if (initial_cache.find(m) != initial_cache.end()) {
                continue;
            }
            initial_cache.emplace(m, build_initial(m));
        }

        struct RunOut {
            int run = 0;
            std::string init_method;
            std::uint64_t seed = 0;
            santa2025::HHResult res;
            std::string error;
        };

        auto run_one = [&](int run_id) -> RunOut {
            RunOut out;
            out.run = run_id;
            out.init_method = init_methods[static_cast<size_t>(run_id) % init_methods.size()];
            constexpr std::uint64_t kSeedStride = 1'000'003ULL;
            out.seed = args.seed + static_cast<std::uint64_t>(run_id) * kSeedStride;

            const auto& initial = initial_cache.at(out.init_method);

            santa2025::HHOptions hh;
            hh.n = args.n;
            hh.seed = out.seed;
            hh.hh_iters = args.hh_iters;
            hh.lahc_length = args.lahc_length;
            hh.ucb_c = args.ucb_c;
            hh.log_every = args.log_every;
            hh.log_prefix = "[run " + std::to_string(run_id) + " hh]";

            if (args.objective == "s") {
                hh.objective = santa2025::HHObjective::kS;
            } else if (args.objective == "score") {
                hh.objective = santa2025::HHObjective::kPrefixScore;
            } else {
                throw std::runtime_error("invalid --objective (use s|score)");
            }
            hh.nmax_score = args.nmax_score;

            // SA base.
            hh.sa_base.n = args.n;
            hh.sa_base.angles_deg = args.angles;
            hh.sa_base.cycle_deg = args.cycle;
            hh.sa_base.orientation_mode = mode;
            hh.sa_base.cycle_prefix = args.cycle_prefix;
            hh.sa_base.iters = args.sa_block_iters;
            hh.sa_base.tries_per_iter = args.sa_tries;
            hh.sa_base.t0 = args.sa_t0;
            hh.sa_base.t1 = args.sa_t1;
            hh.sa_base.trans_sigma0 = args.sa_sigma0;
            hh.sa_base.trans_sigma1 = args.sa_sigma1;
            hh.sa_base.gap = args.gap;
            hh.sa_base.max_offsets_per_delta = args.max_offsets;

            // LNS base.
            hh.lns_base.n = args.n;
            hh.lns_base.angles_deg = args.angles;
            hh.lns_base.cycle_deg = args.cycle;
            hh.lns_base.orientation_mode = mode;
            hh.lns_base.cycle_prefix = args.cycle_prefix;
            hh.lns_base.stages = args.lns_stages;
            hh.lns_base.stage_attempts = args.lns_stage_attempts;
            hh.lns_base.remove_frac = args.lns_remove_frac;
            hh.lns_base.boundary_prob = args.lns_boundary_prob;
            hh.lns_base.shrink_factor = args.lns_shrink_factor;
            hh.lns_base.shrink_delta = args.lns_shrink_delta;
            hh.lns_base.slide_iters = args.lns_slide_iters;
            hh.lns_base.gap = args.gap;
            hh.lns_base.max_offsets_per_delta = args.max_offsets;

            try {
                out.res = santa2025::hyper_heuristic_optimize(poly, initial, hh, args.eps);
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
            if (a.res.best_obj != b.res.best_obj) {
                return a.res.best_obj < b.res.best_obj;
            }
            if (a.res.best_s != b.res.best_s) {
                return a.res.best_s < b.res.best_s;
            }
            if (a.res.best_score != b.res.best_score) {
                return a.res.best_score < b.res.best_score;
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

        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\n";
        out << "  \"multi_start\": {\"runs\": " << args.runs << ", \"threads\": " << args.threads << ", \"best_run\": "
            << best_run << "},\n";
        out << "  \"init\": {\"obj\": " << best.res.init_obj << ", \"s\": " << best.res.init_s << ", \"score\": "
            << best.res.init_score << "},\n";
        out << "  \"best\": {\"obj\": " << best.res.best_obj << ", \"s\": " << best.res.best_s << ", \"score\": "
            << best.res.best_score << "},\n";
        out << "  \"counters\": {\"attempted\": " << best.res.attempted << ", \"feasible\": " << best.res.feasible
            << ", \"accepted\": " << best.res.accepted << "},\n";
        out << "  \"ops\": [\n";
        for (size_t i = 0; i < best.res.ops.size(); ++i) {
            const auto& s = best.res.ops[i];
            out << "    {\"name\": \"" << s.name << "\", \"selected\": " << s.selected << ", \"feasible\": " << s.feasible
                << ", \"accepted\": " << s.accepted << ", \"mean_reward\": " << s.mean_reward << "}";
            if (i + 1 != best.res.ops.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ],\n";
        out << "  \"runs\": [\n";
        for (size_t i = 0; i < runs.size(); ++i) {
            const auto& r = runs[i];
            out << "    {\"run\": " << r.run << ", \"init\": \"" << r.init_method << "\", \"seed\": " << r.seed
                << ", \"best\": {\"obj\": " << r.res.best_obj << ", \"s\": " << r.res.best_s << ", \"score\": "
                << r.res.best_score << "}}";
            if (i + 1 != runs.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ],\n";
        out << "  \"poses\": [\n";
        for (size_t i = 0; i < best.res.best_poses.size(); ++i) {
            const auto& p = best.res.best_poses[i];
            out << "    {\"i\": " << i << ", \"x\": " << p.x << ", \"y\": " << p.y << ", \"deg\": " << p.deg << "}";
            if (i + 1 != best.res.best_poses.size()) {
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
            const int csv_nmax = (args.csv_nmax > 0) ? args.csv_nmax : args.n;
            if (csv_nmax <= 0 || csv_nmax > args.n) {
                throw std::runtime_error("--csv-nmax must be in [1,--n]");
            }
            std::ofstream f(args.out_csv);
            if (!f) {
                throw std::runtime_error("failed to open --out-csv file: " + args.out_csv);
            }
            santa2025::write_prefix_submission_csv(f, best.res.best_poses, csv_nmax, args.csv_precision);
            if (args.log_every > 0) {
                std::lock_guard<std::mutex> lk(santa2025::log_mutex());
                std::cerr << "[hh_opt] wrote submission csv: " << args.out_csv << " (nmax=" << csv_nmax
                          << ", precision=" << args.csv_precision << ")\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

