#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "santa2025/constraints.hpp"
#include "santa2025/geometry.hpp"
#include "santa2025/submission_csv.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    std::string path;
    int nmax = 200;
    double eps = 1e-12;
    bool require_complete = true;
    bool overlap_check = true;
    double min_sep = 0.0;
    bool breakdown = false;
};

long double dist2_point_segment_ld(
    const santa2025::Point& p,
    const santa2025::Point& a,
    const santa2025::Point& b
) {
    const long double ax = static_cast<long double>(a.x);
    const long double ay = static_cast<long double>(a.y);
    const long double bx = static_cast<long double>(b.x);
    const long double by = static_cast<long double>(b.y);
    const long double px = static_cast<long double>(p.x);
    const long double py = static_cast<long double>(p.y);

    const long double vx = bx - ax;
    const long double vy = by - ay;
    const long double wx = px - ax;
    const long double wy = py - ay;

    const long double vv = vx * vx + vy * vy;
    if (!(vv > 0.0L)) {
        return wx * wx + wy * wy;
    }
    long double t = (wx * vx + wy * vy) / vv;
    t = std::clamp(t, 0.0L, 1.0L);

    const long double cx = ax + t * vx;
    const long double cy = ay + t * vy;
    const long double dx = px - cx;
    const long double dy = py - cy;
    return dx * dx + dy * dy;
}

long double polygons_min_sep2_ld(const santa2025::Polygon& a, const santa2025::Polygon& b) {
    if (a.empty() || b.empty()) {
        return 0.0L;
    }
    long double best = std::numeric_limits<long double>::infinity();

    for (size_t i = 0; i < a.size(); ++i) {
        const auto& p = a[i];
        for (size_t j = 0; j < b.size(); ++j) {
            const auto& q1 = b[j];
            const auto& q2 = b[(j + 1) % b.size()];
            best = std::min(best, dist2_point_segment_ld(p, q1, q2));
        }
    }
    for (size_t i = 0; i < b.size(); ++i) {
        const auto& p = b[i];
        for (size_t j = 0; j < a.size(); ++j) {
            const auto& q1 = a[j];
            const auto& q2 = a[(j + 1) % a.size()];
            best = std::min(best, dist2_point_segment_ld(p, q1, q2));
        }
    }

    return best;
}

double aabb_distance(const santa2025::BoundingBox& a, const santa2025::BoundingBox& b) {
    double dx = 0.0;
    if (a.max_x < b.min_x) {
        dx = b.min_x - a.max_x;
    } else if (b.max_x < a.min_x) {
        dx = a.min_x - b.max_x;
    }

    double dy = 0.0;
    if (a.max_y < b.min_y) {
        dy = b.min_y - a.max_y;
    } else if (b.max_y < a.min_y) {
        dy = a.min_y - b.max_y;
    }

    return std::hypot(dx, dy);
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
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "--allow-partial") {
            args.require_complete = false;
        } else if (a == "--no-overlap-check") {
            args.overlap_check = false;
        } else if (a == "--min-sep") {
            args.min_sep = std::stod(need("--min-sep"));
        } else if (a == "--breakdown") {
            args.breakdown = true;
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: score_submission <submission.csv> [--nmax 200] [--eps 1e-12]\n"
                      << "                       [--allow-partial] [--no-overlap-check] [--min-sep g] [--breakdown]\n";
            std::exit(0);
        } else if (!a.empty() && a[0] == '-') {
            throw std::runtime_error("unknown arg: " + a);
        } else if (args.path.empty()) {
            args.path = a;
        } else {
            throw std::runtime_error("unexpected extra arg: " + a);
        }
    }
    if (args.path.empty()) {
        throw std::runtime_error("missing <submission.csv>");
    }
    if (args.nmax <= 0 || args.nmax > 200) {
        throw std::runtime_error("--nmax must be in [1,200]");
    }
    if (!(args.eps > 0.0)) {
        throw std::runtime_error("--eps must be > 0");
    }
    if (!(args.min_sep >= 0.0)) {
        throw std::runtime_error("--min-sep must be >= 0");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);

        std::ifstream f(args.path);
        if (!f) {
            throw std::runtime_error("failed to open: " + args.path);
        }

        santa2025::ReadSubmissionOptions ropt;
        ropt.nmax = args.nmax;
        ropt.require_complete = args.require_complete;
        const auto sub = santa2025::read_submission_csv(f, ropt);

        const santa2025::Polygon tree = santa2025::tree_polygon();

        double total_score = 0.0;
        double max_s = 0.0;

        struct PerN {
            int puzzle = 0;
            double s = 0.0;
            double contrib = 0.0;
        };
        std::vector<PerN> per_n;
        per_n.reserve(static_cast<size_t>(args.nmax));

        for (int puzzle = 1; puzzle <= args.nmax; ++puzzle) {
            const auto& poses = sub.poses[static_cast<size_t>(puzzle)];
            if (poses.empty()) {
                continue;
            }
            if (static_cast<int>(poses.size()) != puzzle) {
                throw std::runtime_error("puzzle " + std::to_string(puzzle) + ": expected " +
                                         std::to_string(puzzle) + " poses");
            }

            for (int i = 0; i < puzzle; ++i) {
                const auto& p = poses[static_cast<size_t>(i)];
                if (!santa2025::within_coord_bounds(p, santa2025::kCoordMin, santa2025::kCoordMax, args.eps)) {
                    throw std::runtime_error("puzzle " + std::to_string(puzzle) + " idx " + std::to_string(i) +
                                             ": (x,y) out of bounds [-100,100]");
                }
            }

            std::vector<santa2025::Polygon> polys;
            std::vector<santa2025::BoundingBox> bbs;
            polys.reserve(static_cast<size_t>(puzzle));
            bbs.reserve(static_cast<size_t>(puzzle));

            for (int i = 0; i < puzzle; ++i) {
                const auto& p = poses[static_cast<size_t>(i)];
                santa2025::Polygon poly = santa2025::translate_polygon(santa2025::rotate_polygon(tree, p.deg), p.x, p.y);
                bbs.push_back(santa2025::polygon_bbox(poly));
                polys.push_back(std::move(poly));
            }

            double min_x = std::numeric_limits<double>::infinity();
            double min_y = std::numeric_limits<double>::infinity();
            double max_x = -std::numeric_limits<double>::infinity();
            double max_y = -std::numeric_limits<double>::infinity();
            for (const auto& bb : bbs) {
                min_x = std::min(min_x, bb.min_x);
                min_y = std::min(min_y, bb.min_y);
                max_x = std::max(max_x, bb.max_x);
                max_y = std::max(max_y, bb.max_y);
            }

            const double width = max_x - min_x;
            const double height = max_y - min_y;
            const double s = std::max(width, height);
            const double contrib = (s * s) / static_cast<double>(puzzle);

            if (args.overlap_check) {
                for (int i = 0; i < puzzle; ++i) {
                    for (int j = i + 1; j < puzzle; ++j) {
                        const auto& a = bbs[static_cast<size_t>(i)];
                        const auto& b = bbs[static_cast<size_t>(j)];
                        if (a.max_x < b.min_x - args.eps || b.max_x < a.min_x - args.eps ||
                            a.max_y < b.min_y - args.eps || b.max_y < a.min_y - args.eps) {
                            continue;
                        }
                        if (santa2025::polygons_overlap_strict(polys[static_cast<size_t>(i)],
                                                             polys[static_cast<size_t>(j)],
                                                             args.eps)) {
                            throw std::runtime_error("overlap detected in puzzle " + std::to_string(puzzle) +
                                                     " between " + std::to_string(i) + " and " + std::to_string(j));
                        }
                    }
                }
            }

            if (args.min_sep > 0.0) {
                const long double gap2 = static_cast<long double>(args.min_sep) * static_cast<long double>(args.min_sep);
                for (int i = 0; i < puzzle; ++i) {
                    for (int j = i + 1; j < puzzle; ++j) {
                        const auto& bb_a = bbs[static_cast<size_t>(i)];
                        const auto& bb_b = bbs[static_cast<size_t>(j)];
                        if (aabb_distance(bb_a, bb_b) >= args.min_sep) {
                            continue;
                        }

                        if (santa2025::polygons_overlap_strict(polys[static_cast<size_t>(i)],
                                                             polys[static_cast<size_t>(j)],
                                                             args.eps)) {
                            throw std::runtime_error("min-sep violated (overlap) in puzzle " + std::to_string(puzzle) +
                                                     " between " + std::to_string(i) + " and " + std::to_string(j));
                        }

                        const long double d2 =
                            polygons_min_sep2_ld(polys[static_cast<size_t>(i)], polys[static_cast<size_t>(j)]);
                        if (d2 < gap2) {
                            const long double d = std::sqrt(std::max(0.0L, d2));
                            std::ostringstream oss;
                            oss << std::setprecision(17);
                            oss << "min-sep violated in puzzle " << puzzle << " between " << i << " and " << j
                                << " dist=" << static_cast<double>(d) << " min_sep=" << args.min_sep;
                            throw std::runtime_error(oss.str());
                        }
                    }
                }
            }

            total_score += contrib;
            max_s = std::max(max_s, s);
            per_n.push_back(PerN{puzzle, s, contrib});
        }

        std::cout << std::setprecision(17);
        std::cout << "{\n";
        std::cout << "  \"nmax\": " << args.nmax << ",\n";
        std::cout << "  \"score\": " << total_score << ",\n";
        std::cout << "  \"s_max\": " << max_s << ",\n";
        std::cout << "  \"overlap_check\": " << (args.overlap_check ? "true" : "false") << ",\n";
        std::cout << "  \"require_complete\": " << (args.require_complete ? "true" : "false");
        if (args.breakdown) {
            std::cout << ",\n  \"per_n\": [\n";
            for (size_t i = 0; i < per_n.size(); ++i) {
                const auto& r = per_n[i];
                std::cout << "    {\"puzzle\": " << r.puzzle << ", \"s\": " << r.s << ", \"contrib\": " << r.contrib
                          << "}";
                if (i + 1 != per_n.size()) {
                    std::cout << ",";
                }
                std::cout << "\n";
            }
            std::cout << "  ]\n";
        } else {
            std::cout << "\n";
        }
        std::cout << "}\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
