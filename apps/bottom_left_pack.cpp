#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "santa2025/bottom_left.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    int n = 200;
    std::vector<double> angles{45.0};
    std::vector<double> cycle{};
    std::string mode = "all";
    int cycle_prefix = 0;
    int max_offsets = 512;
    double gap = 1e-6;
    int slide_iters = 32;
    double eps = 1e-12;
    double side = 0.0;
    double density = 0.4;
    double grow = 1.05;
    int restarts = 40;
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

        if (a == "--n") {
            args.n = std::stoi(need("--n"));
        } else if (a == "--angles") {
            args.angles = parse_angles(need("--angles"));
        } else if (a == "--cycle") {
            args.cycle = parse_angles(need("--cycle"));
        } else if (a == "--mode") {
            args.mode = need("--mode");
        } else if (a == "--cycle-prefix") {
            args.cycle_prefix = std::stoi(need("--cycle-prefix"));
        } else if (a == "--max-offsets") {
            args.max_offsets = std::stoi(need("--max-offsets"));
        } else if (a == "--gap") {
            args.gap = std::stod(need("--gap"));
        } else if (a == "--slide-iters") {
            args.slide_iters = std::stoi(need("--slide-iters"));
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "--side") {
            args.side = std::stod(need("--side"));
        } else if (a == "--density") {
            args.density = std::stod(need("--density"));
        } else if (a == "--grow") {
            args.grow = std::stod(need("--grow"));
        } else if (a == "--restarts") {
            args.restarts = std::stoi(need("--restarts"));
        } else if (a == "-h" || a == "--help") {
            std::cout
                << "Usage: bottom_left_pack [--n N] [--angles a,b,c] [--max-offsets N] [--gap g] [--slide-iters N]\n"
                << "                       [--cycle a,b,c] [--mode all|cycle|cycle-then-all] [--cycle-prefix N]\n"
                << "                       [--side S] [--density d] [--grow f] [--restarts k] [--eps e]\n";
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

        santa2025::BottomLeftOptions opt;
        opt.n = args.n;
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

        const santa2025::Polygon poly = santa2025::tree_polygon();
        const auto poses = santa2025::bottom_left_pack(poly, opt, args.eps);
        const auto stats = santa2025::packing_stats(poly, poses);

        std::cout << std::setprecision(17);
        std::cout << "{\n";
        std::cout << "  \"n\": " << poses.size() << ",\n";
        std::cout << "  \"bbox\": {\"min_x\": " << stats.bbox.min_x << ", \"min_y\": " << stats.bbox.min_y
                  << ", \"max_x\": " << stats.bbox.max_x << ", \"max_y\": " << stats.bbox.max_y << "},\n";
        std::cout << "  \"square_side\": " << stats.square_side << ",\n";
        std::cout << "  \"poses\": [\n";
        for (size_t i = 0; i < poses.size(); ++i) {
            const auto& p = poses[i];
            std::cout << "    {\"i\": " << i << ", \"x\": " << p.x << ", \"y\": " << p.y << ", \"deg\": " << p.deg
                      << "}";
            if (i + 1 != poses.size()) {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
