#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "santa2025/order_hyper_heuristic.hpp"
#include "santa2025/submission_csv.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    std::string in_csv;
    std::string out_csv;
    std::string out_json;

    int puzzle = 200;
    int nmax = 200;

    int iters = 20'000;
    int lahc_length = 0;
    double ucb_c = 0.25;
    std::uint64_t seed = 1;
    int log_every = 0;

    int csv_precision = 17;
    double eps = 1e-12;
};

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

        if (a == "--in") {
            args.in_csv = need("--in");
        } else if (a == "--out") {
            args.out_csv = need("--out");
        } else if (a == "--out-json") {
            args.out_json = need("--out-json");
        } else if (a == "--puzzle") {
            args.puzzle = std::stoi(need("--puzzle"));
        } else if (a == "--nmax") {
            args.nmax = std::stoi(need("--nmax"));
        } else if (a == "--iters") {
            args.iters = std::stoi(need("--iters"));
        } else if (a == "--lahc-length") {
            args.lahc_length = std::stoi(need("--lahc-length"));
        } else if (a == "--ucb-c") {
            args.ucb_c = std::stod(need("--ucb-c"));
        } else if (a == "--seed") {
            args.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
        } else if (a == "--log-every") {
            args.log_every = std::stoi(need("--log-every"));
        } else if (a == "--csv-precision") {
            args.csv_precision = std::stoi(need("--csv-precision"));
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: order_opt --in <submission.csv> --out <submission.csv>\n"
                      << "                 [--puzzle 200] [--nmax 200] [--iters 20000]\n"
                      << "                 [--lahc-length 0] [--ucb-c 0.25] [--seed 1]\n"
                      << "                 [--out-json path] [--log-every 0] [--csv-precision 17] [--eps 1e-12]\n";
            std::exit(0);
        } else if (!a.empty() && a[0] == '-') {
            throw std::runtime_error("unknown arg: " + a);
        } else if (args.in_csv.empty()) {
            args.in_csv = a;
        } else if (args.out_csv.empty()) {
            args.out_csv = a;
        } else {
            throw std::runtime_error("unexpected extra arg: " + a);
        }
    }
    if (args.in_csv.empty()) {
        throw std::runtime_error("missing --in <submission.csv>");
    }
    if (args.out_csv.empty()) {
        throw std::runtime_error("missing --out <submission.csv>");
    }
    if (args.puzzle <= 0 || args.puzzle > 200) {
        throw std::runtime_error("--puzzle must be in [1,200]");
    }
    if (args.nmax <= 0 || args.nmax > args.puzzle) {
        throw std::runtime_error("--nmax must be in [1,--puzzle]");
    }
    if (args.iters <= 0) {
        throw std::runtime_error("--iters must be > 0");
    }
    if (!(args.ucb_c >= 0.0)) {
        throw std::runtime_error("--ucb-c must be >= 0");
    }
    if (!(args.eps > 0.0)) {
        throw std::runtime_error("--eps must be > 0");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);

        std::ifstream f(args.in_csv);
        if (!f) {
            throw std::runtime_error("failed to open: " + args.in_csv);
        }

        santa2025::ReadSubmissionOptions ropt;
        ropt.nmax = 200;
        ropt.require_complete = true;
        const auto sub = santa2025::read_submission_csv(f, ropt);

        const auto& poses = sub.poses[static_cast<size_t>(args.puzzle)];
        if (static_cast<int>(poses.size()) != args.puzzle) {
            throw std::runtime_error("input submission: puzzle " + std::to_string(args.puzzle) + " is missing/invalid");
        }

        santa2025::OrderHHOptions opt;
        opt.n = args.puzzle;
        opt.nmax_score = args.nmax;
        opt.iters = args.iters;
        opt.lahc_length = args.lahc_length;
        opt.ucb_c = args.ucb_c;
        opt.seed = args.seed;
        opt.log_every = args.log_every;

        const santa2025::Polygon tree = santa2025::tree_polygon();
        const auto res = santa2025::optimize_prefix_order_hh(tree, poses, opt, args.eps);

        std::ofstream out(args.out_csv);
        if (!out) {
            throw std::runtime_error("failed to open: " + args.out_csv);
        }
        santa2025::write_prefix_submission_csv(out, res.best_poses, args.nmax, args.csv_precision);

        std::ostringstream payload;
        payload << std::setprecision(17);
        payload << "{\n";
        payload << "  \"puzzle\": " << args.puzzle << ",\n";
        payload << "  \"nmax\": " << args.nmax << ",\n";
        payload << "  \"init_score\": " << res.init_score << ",\n";
        payload << "  \"best_score\": " << res.best_score << ",\n";
        payload << "  \"attempted\": " << res.attempted << ",\n";
        payload << "  \"accepted\": " << res.accepted << ",\n";
        payload << "  \"ops\": [\n";
        for (size_t i = 0; i < res.ops.size(); ++i) {
            const auto& s = res.ops[i];
            payload << "    {\"name\": \"" << s.name << "\", \"selected\": " << s.selected << ", \"accepted\": "
                    << s.accepted << ", \"mean_reward\": " << s.mean_reward << "}";
            if (i + 1 != res.ops.size()) {
                payload << ",";
            }
            payload << "\n";
        }
        payload << "  ]\n";
        payload << "}\n";

        std::cout << payload.str();

        if (!args.out_json.empty()) {
            std::ofstream jf(args.out_json);
            if (!jf) {
                throw std::runtime_error("failed to open --out-json: " + args.out_json);
            }
            jf << payload.str();
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

