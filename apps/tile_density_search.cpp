#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry/geom.hpp"
#include "solvers/solver_tile_options.hpp"
#include "solvers/tiling_pool.hpp"

namespace {

struct Result {
    int k = 0;
    uint64_t seed = 0;
    Pattern pattern;
    double spacing = std::numeric_limits<double>::infinity();
    double area_per_tree = std::numeric_limits<double>::infinity();
    bool ok = false;
};

double area_per_tree_for(const Pattern& pattern, int k, double spacing) {
    if (k <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    const double th = pattern.lattice_theta_deg * 3.14159265358979323846 / 180.0;
    const double cell_area = (spacing * spacing) * pattern.lattice_v_ratio * std::abs(std::sin(th));
    return cell_area / static_cast<double>(k);
}

std::vector<int> parse_k_list(const std::string& s) {
    std::vector<int> ks;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        int k = std::stoi(item);
        ks.push_back(k);
    }
    if (ks.empty()) {
        throw std::runtime_error("--k-list vazio.");
    }
    for (int k : ks) {
        if (k <= 0) {
            throw std::runtime_error("--k-list contém k <= 0.");
        }
        if (k > 50) {
            throw std::runtime_error("--k-list contém k muito grande (use <= 50).");
        }
    }
    return ks;
}

Options parse_args(int argc, char** argv, std::vector<int>& ks, uint64_t& seed0, int& runs) {
    Options opt;
    opt.tile_obj = TileObjective::kDensity;
    opt.tile_iters = 20000;
    opt.tile_opt_lattice = true;

    seed0 = 1;
    runs = 10;
    ks = {2, 3, 4};

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

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Faltou valor para " + name);
            }
            return argv[++i];
        };

        if (arg == "--tile-iters") {
            opt.tile_iters = parse_int(need(arg));
        } else if (arg == "--no-tile-opt-lattice") {
            opt.tile_opt_lattice = false;
        } else if (arg == "--seed0") {
            seed0 = parse_u64(need(arg));
        } else if (arg == "--runs") {
            runs = parse_int(need(arg));
        } else if (arg == "--k-list") {
            ks = parse_k_list(need(arg));
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("Argumento desconhecido: " + arg);
        }
    }

    if (opt.tile_iters <= 0) {
        throw std::runtime_error("--tile-iters precisa ser > 0.");
    }
    if (runs <= 0) {
        throw std::runtime_error("--runs precisa ser > 0.");
    }
    return opt;
}

Result run_one(const Polygon& base_poly,
               double radius,
               int k,
               uint64_t seed,
               const Options& base_opt) {
    Options opt = base_opt;
    opt.seed = seed;

    Pattern pattern = make_initial_pattern(k);
    Pattern best = optimize_tile_by_spacing(base_poly, pattern, radius, opt);

    const double eps = 1e-9;
    const double spacing =
        find_min_safe_spacing(base_poly, best, radius, eps, 2.0 * radius);
    if (!std::isfinite(spacing)) {
        return Result{};
    }

    Result r;
    r.k = k;
    r.seed = seed;
    r.pattern = best;
    r.spacing = spacing;
    r.area_per_tree = area_per_tree_for(best, k, spacing);
    r.ok = std::isfinite(r.area_per_tree);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<int> ks;
        uint64_t seed0 = 1;
        int runs = 10;
        Options opt = parse_args(argc, argv, ks, seed0, runs);

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);

        std::vector<Result> results;
        results.reserve(static_cast<size_t>(runs * static_cast<int>(ks.size())));

        for (int run = 0; run < runs; ++run) {
            uint64_t seed = seed0 + static_cast<uint64_t>(run);
            for (int k : ks) {
                Result r = run_one(base_poly, radius, k, seed, opt);
                if (r.ok) {
                    results.push_back(r);
                }
                std::cout << "run=" << run
                          << " seed=" << seed
                          << " k=" << k;
                if (r.ok) {
                    std::cout << " spacing=" << std::fixed << std::setprecision(9) << r.spacing
                              << " v_ratio=" << std::setprecision(6) << r.pattern.lattice_v_ratio
                              << " theta=" << std::setprecision(3) << r.pattern.lattice_theta_deg
                              << " area_per_tree=" << std::setprecision(9) << r.area_per_tree;
                } else {
                    std::cout << " (failed)";
                }
                std::cout << "\n";
            }
        }

        if (results.empty()) {
            throw std::runtime_error("Nenhum pattern válido encontrado.");
        }

        std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
            if (a.area_per_tree != b.area_per_tree) {
                return a.area_per_tree < b.area_per_tree;
            }
            if (a.k != b.k) {
                return a.k < b.k;
            }
            return a.seed < b.seed;
        });

        const Result& best = results.front();
        std::cout << "BEST"
                  << " k=" << best.k
                  << " seed=" << best.seed
                  << " spacing=" << std::fixed << std::setprecision(9) << best.spacing
                  << " v_ratio=" << std::setprecision(6) << best.pattern.lattice_v_ratio
                  << " theta=" << std::setprecision(3) << best.pattern.lattice_theta_deg
                  << " area_per_tree=" << std::setprecision(9) << best.area_per_tree
                  << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}

