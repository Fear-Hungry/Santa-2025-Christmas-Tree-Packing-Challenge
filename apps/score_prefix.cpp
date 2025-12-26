#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "santa2025/bottom_left.hpp"
#include "santa2025/grid_shake.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    std::string method = "bottom-left";  // bottom-left | grid-shake
    int nmax = 200;
    bool breakdown = false;

    std::vector<double> angles{45.0};
    std::vector<double> cycle{};
    std::string mode = "all";
    int cycle_prefix = 0;

    // Shared-ish.
    double gap = 1e-6;
    int slide_iters = 32;
    double eps = 1e-12;
    double side = 0.0;
    double grow = 1.05;
    int restarts = 40;

    // Bottom-left.
    int max_offsets = 512;
    double density = 0.4;

    // Grid-shake.
    double step = 0.0;
    int passes = 10;
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

        if (a == "--method") {
            args.method = need("--method");
        } else if (a == "--nmax") {
            args.nmax = std::stoi(need("--nmax"));
        } else if (a == "--breakdown") {
            args.breakdown = true;
        } else if (a == "--angles") {
            args.angles = parse_angles(need("--angles"));
        } else if (a == "--cycle") {
            args.cycle = parse_angles(need("--cycle"));
        } else if (a == "--mode") {
            args.mode = need("--mode");
        } else if (a == "--cycle-prefix") {
            args.cycle_prefix = std::stoi(need("--cycle-prefix"));
        } else if (a == "--gap") {
            args.gap = std::stod(need("--gap"));
        } else if (a == "--slide-iters") {
            args.slide_iters = std::stoi(need("--slide-iters"));
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "--side") {
            args.side = std::stod(need("--side"));
        } else if (a == "--grow") {
            args.grow = std::stod(need("--grow"));
        } else if (a == "--restarts") {
            args.restarts = std::stoi(need("--restarts"));
        } else if (a == "--max-offsets") {
            args.max_offsets = std::stoi(need("--max-offsets"));
        } else if (a == "--density") {
            args.density = std::stod(need("--density"));
        } else if (a == "--step") {
            args.step = std::stod(need("--step"));
        } else if (a == "--passes") {
            args.passes = std::stoi(need("--passes"));
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: score_prefix [--method bottom-left|grid-shake] [--nmax N] [--breakdown]\n"
                      << "                    [--angles a,b,c] [--cycle a,b,c] [--mode all|cycle|cycle-then-all]\n"
                      << "                    [--cycle-prefix N] [--gap g] [--slide-iters N] [--side S] [--grow f]\n"
                      << "                    [--restarts k] [--eps e]\n"
                      << "                    [--max-offsets N] [--density d]        (bottom-left)\n"
                      << "                    [--step s] [--passes k]                (grid-shake)\n";
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
        if (args.nmax <= 0 || args.nmax > 200) {
            throw std::runtime_error("--nmax must be in [1,200]");
        }

        const santa2025::Polygon poly = santa2025::tree_polygon();
        std::vector<santa2025::Pose> poses;
        poses.reserve(200);

        if (args.method == "bottom-left") {
            santa2025::BottomLeftOptions opt;
            opt.n = 200;
            opt.angles_deg = args.angles;
            opt.cycle_deg = args.cycle;
            opt.cycle_prefix = args.cycle_prefix;
            if (args.mode == "all") {
                opt.orientation_mode = santa2025::OrientationMode::kTryAll;
            } else if (args.mode == "cycle") {
                opt.orientation_mode = santa2025::OrientationMode::kCycle;
            } else if (args.mode == "cycle-then-all") {
                opt.orientation_mode = santa2025::OrientationMode::kCycleThenAll;
            } else {
                throw std::runtime_error("invalid --mode (use all|cycle|cycle-then-all)");
            }
            opt.max_offsets_per_delta = args.max_offsets;
            opt.gap = args.gap;
            opt.slide_iters = args.slide_iters;
            opt.side = args.side;
            opt.density_guess = args.density;
            opt.side_grow = args.grow;
            opt.max_restarts = args.restarts;
            poses = santa2025::bottom_left_pack(poly, opt, args.eps);
        } else if (args.method == "grid-shake") {
            santa2025::GridShakeOptions opt;
            opt.n = 200;
            opt.angles_deg = args.angles;
            opt.cycle_deg = args.cycle;
            opt.cycle_prefix = args.cycle_prefix;
            if (args.mode == "all") {
                opt.orientation_mode = santa2025::OrientationMode::kTryAll;
            } else if (args.mode == "cycle") {
                opt.orientation_mode = santa2025::OrientationMode::kCycle;
            } else if (args.mode == "cycle-then-all") {
                opt.orientation_mode = santa2025::OrientationMode::kCycleThenAll;
            } else {
                throw std::runtime_error("invalid --mode (use all|cycle|cycle-then-all)");
            }
            opt.step = args.step;
            opt.gap = args.gap;
            opt.passes = args.passes;
            opt.slide_iters = args.slide_iters;
            opt.side = args.side;
            opt.side_grow = args.grow;
            opt.max_restarts = args.restarts;
            poses = santa2025::grid_shake_pack(poly, opt, args.eps);
        } else {
            throw std::runtime_error("invalid --method (use bottom-left|grid-shake)");
        }

        // Compute score for prefixes in O(n) (incremental bbox).
        double min_x = std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();

        double total = 0.0;
        double s_last = 0.0;

        std::cout << std::setprecision(17);
        if (args.breakdown) {
            std::cout << "n,s_n,term\n";
        }

        for (int i = 0; i < args.nmax; ++i) {
            const auto& p = poses[static_cast<size_t>(i)];
            const santa2025::Polygon rot = santa2025::rotate_polygon(poly, p.deg);
            const santa2025::BoundingBox bb = santa2025::polygon_bbox(santa2025::translate_polygon(rot, p.x, p.y));
            min_x = std::min(min_x, bb.min_x);
            min_y = std::min(min_y, bb.min_y);
            max_x = std::max(max_x, bb.max_x);
            max_y = std::max(max_y, bb.max_y);
            const double s_n = std::max(max_x - min_x, max_y - min_y);
            const double term = (s_n * s_n) / static_cast<double>(i + 1);
            total += term;
            s_last = s_n;
            if (args.breakdown) {
                std::cout << (i + 1) << "," << s_n << "," << term << "\n";
            }
        }

        if (!args.breakdown) {
            std::cout << "{\n";
            std::cout << "  \"method\": \"" << args.method << "\",\n";
            std::cout << "  \"nmax\": " << args.nmax << ",\n";
            std::cout << "  \"s_nmax\": " << s_last << ",\n";
            std::cout << "  \"score\": " << total << "\n";
            std::cout << "}\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
