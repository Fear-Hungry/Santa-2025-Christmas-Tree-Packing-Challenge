#include <cstdlib>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "santa2025/geometry.hpp"
#include "santa2025/nfp.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    std::uint64_t seed = 0;
    int samples = 2000;
    double xy_range = 1.6;
    std::vector<double> deltas{0.0, 15.0, 30.0, 45.0, 60.0, 90.0};
    double eps = 1e-12;
};

std::vector<double> parse_deltas(const std::string& s) {
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

        if (a == "--seed") {
            args.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
        } else if (a == "--samples") {
            args.samples = std::stoi(need("--samples"));
        } else if (a == "--range") {
            args.xy_range = std::stod(need("--range"));
        } else if (a == "--deltas") {
            args.deltas = parse_deltas(need("--deltas"));
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: nfp_sanity [--seed N] [--samples N] [--range R] [--deltas a,b,c] [--eps E]\n";
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

        const santa2025::Polygon poly = santa2025::tree_polygon();
        santa2025::NFPCache cache(poly, args.eps);

        const santa2025::Polygon base = santa2025::rotate_polygon(poly, 0.0);

        std::mt19937_64 rng(args.seed);
        std::uniform_real_distribution<double> uni(-args.xy_range, args.xy_range);

        for (const double delta : args.deltas) {
            const santa2025::NFP& nfp = cache.get(delta);
            if (nfp.pieces.empty()) {
                std::cerr << "[FAIL] delta=" << delta << ": NFP has no pieces\n";
                return 2;
            }

            const santa2025::Polygon other = santa2025::rotate_polygon(poly, delta);

            for (int s = 0; s < args.samples; ++s) {
                const double dx = uni(rng);
                const double dy = uni(rng);
                const santa2025::Point t{dx, dy};

                const bool geom_overlap = santa2025::polygons_intersect(
                    base, santa2025::translate_polygon(other, dx, dy), args.eps
                );
                const bool nfp_overlap = santa2025::nfp_contains(nfp, t, args.eps);

                if (geom_overlap != nfp_overlap) {
                    std::cerr << "[FAIL] mismatch\n";
                    std::cerr << "  delta=" << delta << "\n";
                    std::cerr << std::setprecision(17) << "  t=(" << dx << "," << dy << ")\n";
                    std::cerr << "  geom_overlap=" << (geom_overlap ? "true" : "false") << "\n";
                    std::cerr << "  nfp_overlap=" << (nfp_overlap ? "true" : "false") << "\n";
                    return 1;
                }
            }

            std::cout << "[OK] delta=" << delta << " samples=" << args.samples << " pieces=" << nfp.pieces.size()
                      << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
