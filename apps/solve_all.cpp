#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "santa2025/bottom_left.hpp"
#include "santa2025/collision_index.hpp"
#include "santa2025/constraints.hpp"
#include "santa2025/grid_shake.hpp"
#include "santa2025/hyper_heuristic.hpp"
#include "santa2025/logging.hpp"
#include "santa2025/lns.hpp"
#include "santa2025/simulated_annealing.hpp"
#include "santa2025/submission_csv.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    int nmax = 200;

    std::string init = "grid-shake";  // grid-shake | bottom-left
    bool warm_start = true;

    // Refinement per puzzle: none | sa | hh
    std::string refine = "sa";

    // Multi-start per puzzle (independent seeds).
    int runs_per_n = 1;
    int threads = 1;
    std::uint64_t seed = 1;

    // Orientation control.
    std::vector<double> angles{45.0};
    std::vector<double> cycle{};
    std::string mode = "all";  // all | cycle | cycle-then-all
    int cycle_prefix = 0;

    // Shared geometry knobs.
    double gap = 1e-6;
    double safety_eps = 0.0;
    int slide_iters = 32;
    int max_offsets = 512;
    double side_grow = 1.05;
    int max_restarts = 40;

    // SA base budget (scaled per n by --sa-iters-mode).
    int sa_iters = 20'000;
    std::string sa_iters_mode = "linear";  // linear | constant
    int sa_tries = 8;
    double sa_t0 = 0.25;
    double sa_t1 = 1e-4;
    double sa_sigma0 = 0.20;
    double sa_sigma1 = 0.01;

    // Optional LNS post-pass after refine.
    bool lns = false;
    int lns_stages = 3;
    int lns_stage_attempts = 20;
    double lns_remove_frac = 0.12;
    double lns_boundary_prob = 0.7;
    double lns_shrink_factor = 0.999;
    double lns_shrink_delta = 0.0;
    int lns_slide_iters = 60;

    // HH options (used when refine == hh).
    int hh_iters = 60;
    int lahc_length = 0;
    double ucb_c = 0.25;

    // Output.
    std::string out_csv;
    std::string out_dir;
    std::string out_json;
    int csv_precision = 17;

    int log_every = 1;  // per-n progress
    double eps = 1e-12;
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

        if (a == "--nmax") {
            args.nmax = std::stoi(need("--nmax"));
        } else if (a == "--init") {
            args.init = need("--init");
        } else if (a == "--no-warm-start") {
            args.warm_start = false;
        } else if (a == "--refine") {
            args.refine = need("--refine");
        } else if (a == "--runs-per-n") {
            args.runs_per_n = std::stoi(need("--runs-per-n"));
        } else if (a == "--threads") {
            args.threads = std::stoi(need("--threads"));
        } else if (a == "--seed") {
            args.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
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
        } else if (a == "--safety-eps") {
            args.safety_eps = std::stod(need("--safety-eps"));
        } else if (a == "--slide-iters") {
            args.slide_iters = std::stoi(need("--slide-iters"));
        } else if (a == "--max-offsets") {
            args.max_offsets = std::stoi(need("--max-offsets"));
        } else if (a == "--side-grow") {
            args.side_grow = std::stod(need("--side-grow"));
        } else if (a == "--restarts") {
            args.max_restarts = std::stoi(need("--restarts"));
        } else if (a == "--sa-iters") {
            args.sa_iters = std::stoi(need("--sa-iters"));
        } else if (a == "--sa-iters-mode") {
            args.sa_iters_mode = need("--sa-iters-mode");
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
        } else if (a == "--lns") {
            args.lns = true;
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
        } else if (a == "--hh-iters") {
            args.hh_iters = std::stoi(need("--hh-iters"));
        } else if (a == "--lahc-length") {
            args.lahc_length = std::stoi(need("--lahc-length"));
        } else if (a == "--ucb-c") {
            args.ucb_c = std::stod(need("--ucb-c"));
        } else if (a == "--out") {
            args.out_csv = need("--out");
        } else if (a == "--out-dir") {
            args.out_dir = need("--out-dir");
        } else if (a == "--out-json") {
            args.out_json = need("--out-json");
        } else if (a == "--csv-precision") {
            args.csv_precision = std::stoi(need("--csv-precision"));
        } else if (a == "--log-every") {
            args.log_every = std::stoi(need("--log-every"));
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "-h" || a == "--help") {
            std::cout
                << "Usage: solve_all --out submission.csv [--nmax 200] [--init grid-shake|bottom-left]\n"
                << "                [--refine none|sa|hh] [--no-warm-start]\n"
                << "                [--runs-per-n 1] [--threads N] [--seed 1]\n"
                << "                [--angles a,b,c] [--cycle a,b,c] [--mode all|cycle|cycle-then-all]\n"
                << "                [--cycle-prefix N]\n"
                << "                [--sa-iters 20000] [--sa-iters-mode linear|constant] [--sa-tries 8]\n"
                << "                [--sa-t0 0.25] [--sa-t1 1e-4] [--sa-sigma0 0.20] [--sa-sigma1 0.01]\n"
                << "                [--lns] [--lns-stages 3] [--lns-stage-attempts 20] [--lns-remove-frac 0.12]\n"
                << "                [--lns-boundary-prob 0.7] [--lns-shrink-factor 0.999] [--lns-shrink-delta 0]\n"
                << "                [--lns-slide-iters 60]\n"
                << "                [--hh-iters 60] [--lahc-length 0] [--ucb-c 0.25]\n"
                << "                [--gap 1e-6] [--safety-eps 0] [--slide-iters 32] [--max-offsets 512] [--side-grow 1.05] [--restarts 40]\n"
                << "                [--out-dir runs/solve_all] [--out-json run.json] [--csv-precision 17]\n";
            std::exit(0);
        } else if (!a.empty() && a[0] == '-') {
            throw std::runtime_error("unknown arg: " + a);
        } else if (args.out_csv.empty()) {
            args.out_csv = a;
        } else {
            throw std::runtime_error("unexpected extra arg: " + a);
        }
    }

    if (args.nmax <= 0 || args.nmax > 200) {
        throw std::runtime_error("--nmax must be in [1,200]");
    }
    if (args.out_csv.empty()) {
        throw std::runtime_error("missing --out <submission.csv>");
    }
    if (args.angles.empty()) {
        throw std::runtime_error("--angles must be non-empty");
    }
    if (args.mode != "all" && args.mode != "cycle" && args.mode != "cycle-then-all") {
        throw std::runtime_error("invalid --mode (use all|cycle|cycle-then-all)");
    }
    if (args.mode != "all" && args.cycle.empty()) {
        throw std::runtime_error("--cycle must be set when using cycle modes");
    }
    if (args.runs_per_n <= 0) {
        throw std::runtime_error("--runs-per-n must be > 0");
    }
    if (args.threads == 0) {
        args.threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    args.threads = std::max(1, args.threads);
    if (args.sa_iters <= 0) {
        throw std::runtime_error("--sa-iters must be > 0");
    }
    if (args.sa_iters_mode != "linear" && args.sa_iters_mode != "constant") {
        throw std::runtime_error("invalid --sa-iters-mode (use linear|constant)");
    }
    if (args.sa_tries <= 0) {
        throw std::runtime_error("--sa-tries must be > 0");
    }
    if (!(args.sa_t0 > 0.0) || !(args.sa_t1 > 0.0)) {
        throw std::runtime_error("--sa-t0/--sa-t1 must be > 0");
    }
    if (args.hh_iters <= 0) {
        throw std::runtime_error("--hh-iters must be > 0");
    }
    if (args.lahc_length < 0) {
        throw std::runtime_error("--lahc-length must be >= 0");
    }
    if (!(args.ucb_c >= 0.0)) {
        throw std::runtime_error("--ucb-c must be >= 0");
    }
    if (!(args.gap >= 0.0)) {
        throw std::runtime_error("--gap must be >= 0");
    }
    if (!(args.safety_eps >= 0.0)) {
        throw std::runtime_error("--safety-eps must be >= 0");
    }
    if (args.slide_iters <= 0) {
        throw std::runtime_error("--slide-iters must be > 0");
    }
    if (args.max_offsets < 0) {
        throw std::runtime_error("--max-offsets must be >= 0");
    }
    if (!(args.side_grow > 1.0)) {
        throw std::runtime_error("--side-grow must be > 1");
    }
    if (args.max_restarts <= 0) {
        throw std::runtime_error("--restarts must be > 0");
    }
    if (args.csv_precision < 0 || args.csv_precision > 17) {
        throw std::runtime_error("--csv-precision must be in [0,17]");
    }
    if (args.lns) {
        if (args.lns_stages < 0 || args.lns_stage_attempts <= 0) {
            throw std::runtime_error("--lns-stages/--lns-stage-attempts invalid");
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
        if (!(args.lns_shrink_delta >= 0.0)) {
            throw std::runtime_error("--lns-shrink-delta must be >= 0");
        }
        if (args.lns_slide_iters <= 0) {
            throw std::runtime_error("--lns-slide-iters must be > 0");
        }
    }
    if (!(args.eps > 0.0)) {
        throw std::runtime_error("--eps must be > 0");
    }

    return args;
}

struct RotatedBBoxCache {
    std::unordered_map<std::int64_t, santa2025::BoundingBox> bboxes;
};

constexpr double kQuant = 1e6;

std::int64_t quant_deg(double deg) {
    return static_cast<std::int64_t>(std::llround(deg * kQuant));
}

std::uint64_t pack_xy_key(std::int32_t qx, std::int32_t qy) {
    const std::uint64_t ux = static_cast<std::uint32_t>(qx);
    const std::uint64_t uy = static_cast<std::uint32_t>(qy);
    return (ux << 32) | uy;
}

santa2025::BoundingBox rotated_bbox_cached(
    const santa2025::Polygon& base,
    double deg,
    RotatedBBoxCache& cache
) {
    const std::int64_t k = quant_deg(deg);
    auto it = cache.bboxes.find(k);
    if (it != cache.bboxes.end()) {
        return it->second;
    }
    const santa2025::BoundingBox bb = santa2025::polygon_bbox(santa2025::rotate_polygon(base, deg));
    cache.bboxes.emplace(k, bb);
    return bb;
}

santa2025::BoundingBox bbox_for_pose(const santa2025::BoundingBox& local, const santa2025::Pose& p) {
    return santa2025::BoundingBox{
        local.min_x + p.x,
        local.min_y + p.y,
        local.max_x + p.x,
        local.max_y + p.y,
    };
}

struct BoundsMetrics {
    double min_x = 0.0;
    double min_y = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double s = 0.0;
};

BoundsMetrics bounds_metrics(const std::vector<santa2025::BoundingBox>& world_bbs) {
    BoundsMetrics m;
    if (world_bbs.empty()) {
        return m;
    }
    m.min_x = std::numeric_limits<double>::infinity();
    m.min_y = std::numeric_limits<double>::infinity();
    m.max_x = -std::numeric_limits<double>::infinity();
    m.max_y = -std::numeric_limits<double>::infinity();
    for (const auto& bb : world_bbs) {
        m.min_x = std::min(m.min_x, bb.min_x);
        m.min_y = std::min(m.min_y, bb.min_y);
        m.max_x = std::max(m.max_x, bb.max_x);
        m.max_y = std::max(m.max_y, bb.max_y);
    }
    m.width = m.max_x - m.min_x;
    m.height = m.max_y - m.min_y;
    m.s = std::max(m.width, m.height);
    return m;
}

bool within_container_quadrant(const santa2025::BoundingBox& bb, double eps) {
    return bb.min_x >= -eps && bb.min_y >= -eps;
}

bool within_container_square(const santa2025::BoundingBox& bb, double side, double eps) {
    return within_container_quadrant(bb, eps) && (bb.max_x <= side + eps) && (bb.max_y <= side + eps);
}

bool is_valid_pose(
    const santa2025::Pose& p,
    const santa2025::BoundingBox& local_bb,
    const santa2025::CollisionIndex& index,
    double side,
    double safety_eps,
    double eps
) {
    if (!santa2025::within_coord_bounds(p, santa2025::kCoordMin, santa2025::kCoordMax, eps)) {
        return false;
    }
    const santa2025::BoundingBox bb = bbox_for_pose(local_bb, p);
    if (!within_container_square(bb, side, eps)) {
        return false;
    }
    return !index.collides_with_any(p, -1, safety_eps);
}

santa2025::Pose slide_down_left(
    santa2025::Pose p,
    const santa2025::BoundingBox& local_bb,
    const santa2025::CollisionIndex& index,
    double side,
    double safety_eps,
    int iters,
    double eps
) {
    double x_low = -local_bb.min_x;
    double x_high = side - local_bb.max_x;
    double y_low = -local_bb.min_y;
    double y_high = side - local_bb.max_y;

    auto normalize_interval = [&](double& lo, double& hi) -> bool {
        if (lo <= hi) {
            return true;
        }
        if (lo <= hi + eps) {
            const double mid = 0.5 * (lo + hi);
            lo = mid;
            hi = mid;
            return true;
        }
        return false;
    };

    if (!normalize_interval(x_low, x_high) || !normalize_interval(y_low, y_high)) {
        return santa2025::Pose{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), p.deg};
    }

    p.x = std::clamp(p.x, x_low, x_high);
    p.y = std::clamp(p.y, y_low, y_high);

    if (!is_valid_pose(p, local_bb, index, side, safety_eps, eps)) {
        return santa2025::Pose{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), p.deg};
    }

    // Drop down.
    {
        double lo = y_low;
        double hi = p.y;
        for (int k = 0; k < iters; ++k) {
            const double mid = 0.5 * (lo + hi);
            santa2025::Pose tmp = p;
            tmp.y = mid;
            if (is_valid_pose(tmp, local_bb, index, side, safety_eps, eps)) {
                hi = mid;
                p = tmp;
            } else {
                lo = mid;
            }
        }
        p.y = hi;
    }

    // Push left.
    {
        double lo = x_low;
        double hi = p.x;
        for (int k = 0; k < iters; ++k) {
            const double mid = 0.5 * (lo + hi);
            santa2025::Pose tmp = p;
            tmp.x = mid;
            if (is_valid_pose(tmp, local_bb, index, side, safety_eps, eps)) {
                hi = mid;
                p = tmp;
            } else {
                lo = mid;
            }
        }
        p.x = hi;
    }

    return p;
}

std::vector<santa2025::Point> nfp_vertices_sorted_limited(const santa2025::NFP& nfp, int max_n, double eps) {
    std::unordered_set<std::uint64_t> seen;
    std::vector<santa2025::Point> verts;

    for (const auto& piece : nfp.pieces) {
        for (const auto& v : piece) {
            const auto qx = static_cast<std::int32_t>(std::llround(v.x * kQuant));
            const auto qy = static_cast<std::int32_t>(std::llround(v.y * kQuant));
            const std::uint64_t key = pack_xy_key(qx, qy);
            if (seen.insert(key).second) {
                verts.push_back(v);
            }
        }
    }

    std::sort(verts.begin(), verts.end(), [](const santa2025::Point& a, const santa2025::Point& b) {
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });

    if (max_n > 0 && static_cast<int>(verts.size()) > max_n) {
        verts.resize(static_cast<size_t>(max_n));
    }

    verts.erase(std::remove_if(verts.begin(),
                               verts.end(),
                               [&](const santa2025::Point& v) { return std::hypot(v.x, v.y) <= eps; }),
                verts.end());
    return verts;
}

bool normalize_to_quadrant(
    const santa2025::Polygon& tree_poly,
    std::vector<santa2025::Pose>& poses,
    RotatedBBoxCache& bbox_cache,
    double eps
) {
    if (poses.empty()) {
        return true;
    }
    std::vector<santa2025::BoundingBox> world;
    world.reserve(poses.size());
    for (const auto& p : poses) {
        const santa2025::BoundingBox local = rotated_bbox_cached(tree_poly, p.deg, bbox_cache);
        world.push_back(bbox_for_pose(local, p));
    }
    const auto m = bounds_metrics(world);
    const double dx = -m.min_x;
    const double dy = -m.min_y;
    if (std::abs(dx) == 0.0 && std::abs(dy) == 0.0) {
        return true;
    }
    for (const auto& p : poses) {
        if (!santa2025::within_coord_bounds(p.x + dx, p.y + dy, santa2025::kCoordMin, santa2025::kCoordMax, eps)) {
            return false;
        }
    }
    for (auto& p : poses) {
        p.x += dx;
        p.y += dy;
    }
    return true;
}

bool cycle_applies(const Args& args, int i) {
    if (args.cycle.empty()) {
        return false;
    }
    if (args.cycle_prefix <= 0) {
        return true;
    }
    return i < args.cycle_prefix;
}

std::vector<double> angle_candidates(const Args& args, int i) {
    if (args.mode == "all") {
        return args.angles;
    }
    if (!cycle_applies(args, i)) {
        return args.angles;
    }
    const double cyc = args.cycle[static_cast<size_t>(i) % args.cycle.size()];
    if (args.mode == "cycle") {
        return {cyc};
    }
    std::vector<double> out;
    out.reserve(args.angles.size() + 1);
    out.push_back(cyc);
    for (const double a : args.angles) {
        if (std::abs(a - cyc) <= 1e-12) {
            continue;
        }
        out.push_back(a);
    }
    return out;
}

bool insert_one_bottom_left(
    const santa2025::Polygon& tree_poly,
    std::vector<santa2025::Pose>& poses,
    const Args& args,
    double side,
    RotatedBBoxCache& bbox_cache,
    double eps
) {
    const int n = static_cast<int>(poses.size());
    if (n <= 0) {
        return false;
    }
    const int id = n - 1;

    santa2025::CollisionIndex index(tree_poly, eps);
    index.resize(n);
    for (int i = 0; i < id; ++i) {
        index.set_pose(i, poses[static_cast<size_t>(i)]);
    }

    std::unordered_map<std::int64_t, std::vector<santa2025::Point>> verts_cache;

    auto offsets_for = [&](double delta_deg) -> const std::vector<santa2025::Point>& {
        const std::int64_t key = quant_deg(delta_deg);
        auto it = verts_cache.find(key);
        if (it != verts_cache.end()) {
            return it->second;
        }
        const auto& nfp = index.nfp(delta_deg);
        auto verts = nfp_vertices_sorted_limited(nfp, args.max_offsets, eps);
        auto [ins, _ok] = verts_cache.emplace(key, std::move(verts));
        return ins->second;
    };

    santa2025::Pose best{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 0.0};
    bool any_angle = false;

    const auto angs = angle_candidates(args, id);
    for (const double ang : angs) {
        const santa2025::BoundingBox local_bb = rotated_bbox_cached(tree_poly, ang, bbox_cache);
        if (local_bb.max_x - local_bb.min_x > side + eps || local_bb.max_y - local_bb.min_y > side + eps) {
            continue;
        }

        const double x_low = -local_bb.min_x;
        const double y_low = -local_bb.min_y;

        auto consider = [&](santa2025::Pose cand_start) {
            cand_start.deg = ang;
            santa2025::Pose cand =
                slide_down_left(cand_start, local_bb, index, side, args.safety_eps, args.slide_iters, eps);
            if (!std::isfinite(cand.x) || !std::isfinite(cand.y)) {
                return;
            }
            any_angle = true;
            if (cand.y < best.y - 1e-12 || (std::abs(cand.y - best.y) <= 1e-12 && cand.x < best.x)) {
                best = cand;
            }
        };

        // Bottom-left corner.
        consider(santa2025::Pose{x_low, y_low, ang});

        // Touch existing trees.
        for (int j = 0; j < id; ++j) {
            const auto& anchor = poses[static_cast<size_t>(j)];
            const santa2025::BoundingBox anchor_local_bb = rotated_bbox_cached(tree_poly, anchor.deg, bbox_cache);
            const santa2025::BoundingBox anchor_world_bb = bbox_for_pose(anchor_local_bb, anchor);
            const double delta = ang - anchor.deg;

            consider(santa2025::Pose{anchor_world_bb.max_x - local_bb.min_x + args.gap, y_low, ang});
            consider(santa2025::Pose{x_low, anchor_world_bb.max_y - local_bb.min_y + args.gap, ang});
            consider(santa2025::Pose{anchor_world_bb.max_x - local_bb.min_x + args.gap,
                                     anchor_world_bb.max_y - local_bb.min_y + args.gap,
                                     ang});

            const auto& offsets = offsets_for(delta);
            for (const auto& v : offsets) {
                const double nrm = std::hypot(v.x, v.y);
                if (!(nrm > 0.0)) {
                    continue;
                }
                const santa2025::Point u{v.x / nrm, v.y / nrm};
                const santa2025::Point v_out{v.x + args.gap * u.x, v.y + args.gap * u.y};
                const santa2025::Point t_world = santa2025::rotate_point(v_out, anchor.deg);
                consider(santa2025::Pose{anchor.x + t_world.x, anchor.y + t_world.y, ang});
            }
        }
    }

    if (!any_angle || !std::isfinite(best.x)) {
        return false;
    }
    poses[static_cast<size_t>(id)] = best;
    return true;
}

std::vector<santa2025::Pose> build_initial(const santa2025::Polygon& poly, int n, const Args& args, double eps) {
    if (args.init == "grid-shake") {
        santa2025::GridShakeOptions opt;
        opt.n = n;
        opt.angles_deg = args.angles;
        opt.cycle_deg = args.cycle;
        opt.cycle_prefix = args.cycle_prefix;
        if (args.mode == "all") {
            opt.orientation_mode = santa2025::OrientationMode::kTryAll;
        } else if (args.mode == "cycle") {
            opt.orientation_mode = santa2025::OrientationMode::kCycle;
        } else {
            opt.orientation_mode = santa2025::OrientationMode::kCycleThenAll;
        }
        opt.gap = std::max(opt.gap, args.gap);
        opt.safety_eps = args.safety_eps;
        opt.slide_iters = args.slide_iters;
        opt.side_grow = args.side_grow;
        opt.max_restarts = args.max_restarts;
        return santa2025::grid_shake_pack(poly, opt, eps);
    }
    if (args.init == "bottom-left") {
        santa2025::BottomLeftOptions opt;
        opt.n = n;
        opt.angles_deg = args.angles;
        opt.cycle_deg = args.cycle;
        opt.cycle_prefix = args.cycle_prefix;
        if (args.mode == "all") {
            opt.orientation_mode = santa2025::OrientationMode::kTryAll;
        } else if (args.mode == "cycle") {
            opt.orientation_mode = santa2025::OrientationMode::kCycle;
        } else {
            opt.orientation_mode = santa2025::OrientationMode::kCycleThenAll;
        }
        opt.gap = args.gap;
        opt.safety_eps = args.safety_eps;
        opt.slide_iters = args.slide_iters;
        opt.side_grow = args.side_grow;
        opt.max_restarts = args.max_restarts;
        opt.max_offsets_per_delta = args.max_offsets;
        return santa2025::bottom_left_pack(poly, opt, eps);
    }
    throw std::runtime_error("invalid --init (use grid-shake|bottom-left)");
}

int scaled_iters(int n, const Args& args) {
    if (args.sa_iters_mode == "constant") {
        return args.sa_iters;
    }
    const double frac = static_cast<double>(n) / static_cast<double>(args.nmax);
    const int iters = static_cast<int>(std::llround(frac * static_cast<double>(args.sa_iters)));
    return std::max(1, iters);
}

struct SolveOut {
    int n = 0;
    std::uint64_t seed = 0;
    std::vector<santa2025::Pose> best_poses;
    double best_s = 0.0;
    std::string error;
};

SolveOut refine_one_run(
    const santa2025::Polygon& poly,
    int n,
    const std::vector<santa2025::Pose>& initial,
    const Args& args,
    std::uint64_t seed,
    double eps
) {
    SolveOut out;
    out.n = n;
    out.seed = seed;

    std::vector<santa2025::Pose> best = initial;

    if (args.refine == "none") {
        // keep initial
    } else if (args.refine == "sa") {
        santa2025::SAOptions sa;
        sa.n = n;
        sa.angles_deg = args.angles;
        sa.cycle_deg = args.cycle;
        sa.cycle_prefix = args.cycle_prefix;
        if (args.mode == "all") {
            sa.orientation_mode = santa2025::OrientationMode::kTryAll;
        } else if (args.mode == "cycle") {
            sa.orientation_mode = santa2025::OrientationMode::kCycle;
        } else {
            sa.orientation_mode = santa2025::OrientationMode::kCycleThenAll;
        }
        sa.objective = santa2025::SAObjective::kS200;
        sa.seed = seed;
        sa.iters = scaled_iters(n, args);
        sa.tries_per_iter = args.sa_tries;
        sa.t0 = args.sa_t0;
        sa.t1 = args.sa_t1;
        sa.trans_sigma0 = args.sa_sigma0;
        sa.trans_sigma1 = args.sa_sigma1;
        sa.gap = args.gap;
        sa.safety_eps = args.safety_eps;
        sa.max_offsets_per_delta = args.max_offsets;
        sa.log_every = 0;

        const auto res = santa2025::simulated_annealing(poly, best, sa, eps);
        best = res.best_poses;
    } else if (args.refine == "hh") {
        santa2025::HHOptions hh;
        hh.n = n;
        hh.objective = santa2025::HHObjective::kS;
        hh.hh_iters = args.hh_iters;
        hh.lahc_length = args.lahc_length;
        hh.ucb_c = args.ucb_c;
        hh.seed = seed;
        hh.log_every = 0;

        // SA base.
        hh.sa_base.n = n;
        hh.sa_base.angles_deg = args.angles;
        hh.sa_base.cycle_deg = args.cycle;
        hh.sa_base.cycle_prefix = args.cycle_prefix;
        if (args.mode == "all") {
            hh.sa_base.orientation_mode = santa2025::OrientationMode::kTryAll;
        } else if (args.mode == "cycle") {
            hh.sa_base.orientation_mode = santa2025::OrientationMode::kCycle;
        } else {
            hh.sa_base.orientation_mode = santa2025::OrientationMode::kCycleThenAll;
        }
        hh.sa_base.iters = scaled_iters(n, args);
        hh.sa_base.tries_per_iter = args.sa_tries;
        hh.sa_base.t0 = args.sa_t0;
        hh.sa_base.t1 = args.sa_t1;
        hh.sa_base.trans_sigma0 = args.sa_sigma0;
        hh.sa_base.trans_sigma1 = args.sa_sigma1;
        hh.sa_base.gap = args.gap;
        hh.sa_base.safety_eps = args.safety_eps;
        hh.sa_base.max_offsets_per_delta = args.max_offsets;

        // LNS base.
        hh.lns_base.n = n;
        hh.lns_base.angles_deg = args.angles;
        hh.lns_base.cycle_deg = args.cycle;
        hh.lns_base.cycle_prefix = args.cycle_prefix;
        if (args.mode == "all") {
            hh.lns_base.orientation_mode = santa2025::OrientationMode::kTryAll;
        } else if (args.mode == "cycle") {
            hh.lns_base.orientation_mode = santa2025::OrientationMode::kCycle;
        } else {
            hh.lns_base.orientation_mode = santa2025::OrientationMode::kCycleThenAll;
        }
        hh.lns_base.stages = args.lns_stages;
        hh.lns_base.stage_attempts = args.lns_stage_attempts;
        hh.lns_base.remove_frac = args.lns_remove_frac;
        hh.lns_base.boundary_prob = args.lns_boundary_prob;
        hh.lns_base.shrink_factor = args.lns_shrink_factor;
        hh.lns_base.shrink_delta = args.lns_shrink_delta;
        hh.lns_base.slide_iters = args.lns_slide_iters;
        hh.lns_base.gap = args.gap;
        hh.lns_base.safety_eps = args.safety_eps;
        hh.lns_base.max_offsets_per_delta = args.max_offsets;

        const auto res = santa2025::hyper_heuristic_optimize(poly, best, hh, eps);
        best = res.best_poses;
    } else {
        out.error = "invalid --refine (use none|sa|hh)";
        return out;
    }

    if (args.lns) {
        santa2025::LNSOptions lns;
        lns.n = n;
        lns.angles_deg = args.angles;
        lns.cycle_deg = args.cycle;
        lns.cycle_prefix = args.cycle_prefix;
        if (args.mode == "all") {
            lns.orientation_mode = santa2025::OrientationMode::kTryAll;
        } else if (args.mode == "cycle") {
            lns.orientation_mode = santa2025::OrientationMode::kCycle;
        } else {
            lns.orientation_mode = santa2025::OrientationMode::kCycleThenAll;
        }
        lns.stages = args.lns_stages;
        lns.stage_attempts = args.lns_stage_attempts;
        lns.remove_frac = args.lns_remove_frac;
        lns.boundary_prob = args.lns_boundary_prob;
        lns.shrink_factor = args.lns_shrink_factor;
        lns.shrink_delta = args.lns_shrink_delta;
        lns.slide_iters = args.lns_slide_iters;
        lns.gap = args.gap;
        lns.safety_eps = args.safety_eps;
        lns.max_offsets_per_delta = args.max_offsets;
        lns.seed = seed;
        lns.log_every = 0;

        const auto r = santa2025::lns_shrink_wrap(poly, best, lns, eps);
        best = r.best_poses;
    }

    out.best_poses = std::move(best);
    out.best_s = santa2025::packing_s200(poly, out.best_poses);
    return out;
}

SolveOut refine_multi_start(
    const santa2025::Polygon& poly,
    int n,
    const std::vector<santa2025::Pose>& initial,
    const Args& args,
    std::uint64_t seed_base,
    double eps
) {
    if (args.runs_per_n == 1) {
        return refine_one_run(poly, n, initial, args, seed_base, eps);
    }

    struct RunOut {
        SolveOut out;
    };

    std::vector<RunOut> runs(static_cast<size_t>(args.runs_per_n));
    std::atomic<int> next{0};

    int threads = args.threads;
    threads = std::max(1, std::min(threads, args.runs_per_n));

    auto worker = [&]() {
        for (;;) {
            const int id = next.fetch_add(1);
            if (id >= args.runs_per_n) {
                return;
            }
            constexpr std::uint64_t kRunStride = 1'000'003ULL;
            const std::uint64_t seed = seed_base + static_cast<std::uint64_t>(id) * kRunStride;
            runs[static_cast<size_t>(id)].out = refine_one_run(poly, n, initial, args, seed, eps);
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

    auto better = [&](const SolveOut& a, const SolveOut& b) {
        if (!a.error.empty()) {
            return false;
        }
        if (!b.error.empty()) {
            return true;
        }
        if (a.best_s != b.best_s) {
            return a.best_s < b.best_s;
        }
        return a.seed < b.seed;
    };

    int best = 0;
    for (int i = 1; i < args.runs_per_n; ++i) {
        if (better(runs[static_cast<size_t>(i)].out, runs[static_cast<size_t>(best)].out)) {
            best = i;
        }
    }
    return runs[static_cast<size_t>(best)].out;
}

void write_puzzle_csv(
    const std::filesystem::path& path,
    int puzzle,
    const std::vector<santa2025::Pose>& poses,
    int precision
) {
    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("failed to open: " + path.string());
    }
    f << "id,x,y,deg\n";
    for (int i = 0; i < puzzle; ++i) {
        const auto& p = poses[static_cast<size_t>(i)];
        f << santa2025::make_submission_id(puzzle, i) << "," << santa2025::format_submission_value(p.x, precision) << ","
          << santa2025::format_submission_value(p.y, precision) << "," << santa2025::format_submission_value(p.deg, precision)
          << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const santa2025::Polygon poly = santa2025::tree_polygon();

        santa2025::Submission sub;
        sub.poses.resize(static_cast<size_t>(args.nmax + 1));

        std::filesystem::path out_dir;
        if (!args.out_dir.empty()) {
            out_dir = std::filesystem::path(args.out_dir);
            std::filesystem::create_directories(out_dir);
        }

        std::vector<double> per_s(static_cast<size_t>(args.nmax + 1), 0.0);
        std::vector<double> per_term(static_cast<size_t>(args.nmax + 1), 0.0);

        std::vector<santa2025::Pose> prev;
        prev.reserve(static_cast<size_t>(args.nmax));

        constexpr std::uint64_t kNStride = 10'000'019ULL;

        for (int n = 1; n <= args.nmax; ++n) {
            std::vector<santa2025::Pose> initial;

            if (args.warm_start && n > 1 && static_cast<int>(prev.size()) == n - 1) {
                std::vector<santa2025::Pose> cand = prev;
                RotatedBBoxCache bbox_cache;
                if (normalize_to_quadrant(poly, cand, bbox_cache, args.eps)) {
                    cand.push_back(santa2025::Pose{0.0, 0.0, 0.0});

                    // Current side from bbox metrics (after normalization).
                    std::vector<santa2025::BoundingBox> world;
                    world.reserve(static_cast<size_t>(n - 1));
                    for (int i = 0; i < n - 1; ++i) {
                        const auto& p = cand[static_cast<size_t>(i)];
                        const santa2025::BoundingBox local_bb = rotated_bbox_cached(poly, p.deg, bbox_cache);
                        world.push_back(bbox_for_pose(local_bb, p));
                    }
                    const double side0 = bounds_metrics(world).s;

                    bool ok = false;
                    double side = std::max(1e-9, side0);
                    for (int attempt = 0; attempt < args.max_restarts; ++attempt) {
                        if (insert_one_bottom_left(poly, cand, args, side, bbox_cache, args.eps)) {
                            ok = true;
                            break;
                        }
                        side *= args.side_grow;
                    }
                    if (ok) {
                        initial = std::move(cand);
                    }
                }
            }

            if (initial.empty()) {
                initial = build_initial(poly, n, args, args.eps);
            }

            const std::uint64_t seed_n = args.seed + static_cast<std::uint64_t>(n) * kNStride;
            auto best = refine_multi_start(poly, n, initial, args, seed_n, args.eps);
            if (!best.error.empty()) {
                throw std::runtime_error("n=" + std::to_string(n) + " failed: " + best.error);
            }
            if (static_cast<int>(best.best_poses.size()) != n) {
                throw std::runtime_error("n=" + std::to_string(n) + ": solver returned wrong pose count");
            }

            sub.poses[static_cast<size_t>(n)] = best.best_poses;
            prev = best.best_poses;

            const double s = best.best_s;
            const double term = (s * s) / static_cast<double>(n);
            per_s[static_cast<size_t>(n)] = s;
            per_term[static_cast<size_t>(n)] = term;

            if (!args.out_dir.empty()) {
                std::ostringstream name;
                name << "puzzle_" << std::setw(3) << std::setfill('0') << n << ".csv";
                write_puzzle_csv(out_dir / name.str(), n, best.best_poses, args.csv_precision);
            }

            if (args.log_every > 0 && ((n % args.log_every) == 0 || n == args.nmax)) {
                std::lock_guard<std::mutex> lk(santa2025::log_mutex());
                std::cerr << "[solve_all] n=" << n << "/" << args.nmax << " s=" << std::setprecision(17) << s
                          << " term=" << term << " seed=" << best.seed << "\n";
            }
        }

        double total = 0.0;
        for (int n = 1; n <= args.nmax; ++n) {
            total += per_term[static_cast<size_t>(n)];
        }

        {
            std::ofstream f(args.out_csv);
            if (!f) {
                throw std::runtime_error("failed to open --out file: " + args.out_csv);
            }
            santa2025::write_submission_csv(sub, f, args.nmax, args.csv_precision);
        }

        if (!args.out_dir.empty()) {
            std::ofstream f(out_dir / "per_n.csv");
            if (!f) {
                throw std::runtime_error("failed to open per_n.csv in out-dir");
            }
            f << "puzzle,s,term\n";
            f << std::setprecision(17);
            for (int n = 1; n <= args.nmax; ++n) {
                f << n << "," << per_s[static_cast<size_t>(n)] << "," << per_term[static_cast<size_t>(n)] << "\n";
            }
        }

        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\n";
        out << "  \"nmax\": " << args.nmax << ",\n";
        out << "  \"score\": " << total << ",\n";
        out << "  \"out_csv\": " << "\"" << args.out_csv << "\",\n";
        out << "  \"warm_start\": " << (args.warm_start ? "true" : "false") << ",\n";
        out << "  \"refine\": " << "\"" << args.refine << "\",\n";
        out << "  \"runs_per_n\": " << args.runs_per_n << "\n";
        out << "}\n";

        const std::string payload = out.str();
        std::cout << payload;

        if (!args.out_json.empty()) {
            std::ofstream jf(args.out_json);
            if (!jf) {
                throw std::runtime_error("failed to open --out-json: " + args.out_json);
            }
            jf << payload;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
