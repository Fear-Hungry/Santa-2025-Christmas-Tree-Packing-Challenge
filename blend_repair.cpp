#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"
#include "sa.hpp"

namespace {

struct PoseRec {
    double x = std::numeric_limits<double>::quiet_NaN();
    double y = std::numeric_limits<double>::quiet_NaN();
    double deg = std::numeric_limits<double>::quiet_NaN();
};

struct SubmissionData {
    std::string path;
    std::vector<std::vector<PoseRec>> by_n;  // 0..200
};

struct Extents {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

struct Options {
    int n_min = 1;
    int n_max = 200;
    uint64_t seed = 123456789ULL;

    int topk_per_n = 30;
    int blend_iters = 200;
    int boundary_topk = 20;
    int replace_min = 3;
    int replace_max = 16;

    int repair_passes = 400;
    int repair_attempts = 60;
    double repair_step0_frac = 0.02;
    double repair_step_mult = 1.05;
    double repair_noise_frac = 0.01;
    double repair_ddeg_max = 15.0;
    double repair_p_uniform = 0.20;
    double repair_p_rot = 0.50;

    int sa_restarts = 0;
    int sa_iters = 0;
    double sa_w_micro = 1.0;
    double sa_w_swap_rot = 0.25;
    double sa_w_relocate = 0.15;
    double sa_w_block_translate = 0.05;
    double sa_w_block_rotate = 0.02;
    double sa_w_lns = 0.001;
    int sa_block_size = 6;
    int sa_lns_remove = 6;
    int sa_hh_segment = 50;
    double sa_hh_reaction = 0.20;

    // Squeeze (global): comprime layout e repara overlaps.
    int squeeze_tries = 0;
    double squeeze_alpha_min = 0.985;
    double squeeze_alpha_max = 0.9995;
    double squeeze_p_aniso = 0.70;
    int squeeze_repair_passes = 200;

    bool final_rigid = true;
    int output_decimals = 9;
};

struct CandidateSol {
    std::vector<TreePose> poses;
    std::vector<BoundingBox> bbs;
    std::vector<int> boundary_pool;
    double side = std::numeric_limits<double>::infinity();
};

bool parse_submission_line(const std::string& line,
                           std::string& id,
                           std::string& sx,
                           std::string& sy,
                           std::string& sdeg) {
    std::stringstream ss(line);
    if (!std::getline(ss, id, ',')) {
        return false;
    }
    if (!std::getline(ss, sx, ',')) {
        return false;
    }
    if (!std::getline(ss, sy, ',')) {
        return false;
    }
    if (!std::getline(ss, sdeg, ',')) {
        return false;
    }
    return true;
}

double parse_prefixed_value(const std::string& s) {
    if (s.empty() || s[0] != 's') {
        throw std::runtime_error("Valor sem prefixo 's': " + s);
    }
    return std::stod(s.substr(1));
}

std::string fmt_id(int n, int idx) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << n << "_" << idx;
    return oss.str();
}

SubmissionData load_submission(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Erro ao abrir arquivo: " + path);
    }

    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("Arquivo vazio: " + path);
    }

    SubmissionData sub;
    sub.path = path;
    sub.by_n.resize(201);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::string id, sx, sy, sdeg;
        if (!parse_submission_line(line, id, sx, sy, sdeg)) {
            throw std::runtime_error("Linha inválida em " + path + ": " + line);
        }
        if (id.size() < 5 || id[3] != '_') {
            throw std::runtime_error("Formato de id inválido em " + path + ": " + id);
        }

        int n = std::stoi(id.substr(0, 3));
        int idx = std::stoi(id.substr(4));
        if (n < 1 || n > 200) {
            throw std::runtime_error("n fora de [1,200] em " + path + ": " + id);
        }
        if (idx < 0 || idx >= n) {
            throw std::runtime_error("idx fora de [0,n) em " + path + ": " + id);
        }

        const double x = parse_prefixed_value(sx);
        const double y = parse_prefixed_value(sy);
        const double deg = parse_prefixed_value(sdeg);

        if (x < -100.0 || x > 100.0 || y < -100.0 || y > 100.0) {
            throw std::runtime_error("Coordenada fora de [-100,100] em " + path + ": " + id);
        }

        if (sub.by_n[static_cast<size_t>(n)].empty()) {
            sub.by_n[static_cast<size_t>(n)].resize(static_cast<size_t>(n));
        } else if (static_cast<int>(sub.by_n[static_cast<size_t>(n)].size()) != n) {
            throw std::runtime_error("Tamanho inconsistente para n=" + std::to_string(n) +
                                     " em " + path);
        }

        sub.by_n[static_cast<size_t>(n)][static_cast<size_t>(idx)] = PoseRec{x, y, deg};
    }

    for (int n = 1; n <= 200; ++n) {
        if (static_cast<int>(sub.by_n[static_cast<size_t>(n)].size()) != n) {
            throw std::runtime_error("Instância n=" + std::to_string(n) +
                                     " ausente/incompleta em " + path);
        }
        for (int i = 0; i < n; ++i) {
            const auto& rec = sub.by_n[static_cast<size_t>(n)][static_cast<size_t>(i)];
            if (!std::isfinite(rec.x) || !std::isfinite(rec.y) || !std::isfinite(rec.deg)) {
                throw std::runtime_error("Faltou linha para id " + fmt_id(n, i) +
                                         " em " + path);
            }
        }
    }

    return sub;
}

double wrap_deg(double deg) {
    deg = std::fmod(deg, 360.0);
    if (deg <= -180.0) {
        deg += 360.0;
    } else if (deg > 180.0) {
        deg -= 360.0;
    }
    return deg;
}

double quantize_value(double x, int decimals) {
    const std::string s = fmt_submission_value(x, decimals);
    return std::stod(s.substr(1));
}

TreePose quantize_pose(const TreePose& pose, int decimals) {
    return TreePose{quantize_value(pose.x, decimals),
                    quantize_value(pose.y, decimals),
                    quantize_value(wrap_deg(pose.deg), decimals)};
}

std::vector<TreePose> quantize_poses(const std::vector<TreePose>& poses, int decimals) {
    std::vector<TreePose> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back(quantize_pose(p, decimals));
    }
    return out;
}

Extents compute_extents(const std::vector<BoundingBox>& bbs) {
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();
    for (const auto& bb : bbs) {
        e.min_x = std::min(e.min_x, bb.min_x);
        e.max_x = std::max(e.max_x, bb.max_x);
        e.min_y = std::min(e.min_y, bb.min_y);
        e.max_y = std::max(e.max_y, bb.max_y);
    }
    return e;
}

Extents compute_extents_excluding(const std::vector<BoundingBox>& bbs, int skip) {
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < bbs.size(); ++i) {
        if (static_cast<int>(i) == skip) {
            continue;
        }
        const auto& bb = bbs[i];
        e.min_x = std::min(e.min_x, bb.min_x);
        e.max_x = std::max(e.max_x, bb.max_x);
        e.min_y = std::min(e.min_y, bb.min_y);
        e.max_y = std::max(e.max_y, bb.max_y);
    }
    return e;
}

double side_from_extents(const Extents& e) {
    return std::max(e.max_x - e.min_x, e.max_y - e.min_y);
}

Extents merge_extents_bb(const Extents& e, const BoundingBox& bb) {
    Extents out = e;
    out.min_x = std::min(out.min_x, bb.min_x);
    out.max_x = std::max(out.max_x, bb.max_x);
    out.min_y = std::min(out.min_y, bb.min_y);
    out.max_y = std::max(out.max_y, bb.max_y);
    return out;
}

bool aabb_overlap(const BoundingBox& a, const BoundingBox& b) {
    if (a.max_x < b.min_x || b.max_x < a.min_x) {
        return false;
    }
    if (a.max_y < b.min_y || b.max_y < a.min_y) {
        return false;
    }
    return true;
}

std::vector<BoundingBox> bounding_boxes_for_poses(const Polygon& base_poly,
                                                  const std::vector<TreePose>& poses) {
    std::vector<BoundingBox> bbs;
    bbs.reserve(poses.size());
    for (const auto& pose : poses) {
        bbs.push_back(bounding_box(transform_polygon(base_poly, pose)));
    }
    return bbs;
}

std::vector<int> build_extreme_pool(const std::vector<BoundingBox>& bbs, int topk) {
    const int n = static_cast<int>(bbs.size());
    if (n <= 0) {
        return {};
    }
    topk = std::max(1, std::min(topk, n));

    std::vector<int> idx(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        idx[static_cast<size_t>(i)] = i;
    }

    std::vector<char> mark(static_cast<size_t>(n), 0);
    auto add = [&](auto cmp) {
        std::vector<int> v = idx;
        std::sort(v.begin(), v.end(), cmp);
        for (int i = 0; i < topk; ++i) {
            mark[static_cast<size_t>(v[static_cast<size_t>(i)])] = 1;
        }
    };

    add([&](int a, int b) {
        return bbs[static_cast<size_t>(a)].min_x <
               bbs[static_cast<size_t>(b)].min_x;
    });
    add([&](int a, int b) {
        return bbs[static_cast<size_t>(a)].max_x >
               bbs[static_cast<size_t>(b)].max_x;
    });
    add([&](int a, int b) {
        return bbs[static_cast<size_t>(a)].min_y <
               bbs[static_cast<size_t>(b)].min_y;
    });
    add([&](int a, int b) {
        return bbs[static_cast<size_t>(a)].max_y >
               bbs[static_cast<size_t>(b)].max_y;
    });

    std::vector<int> pool;
    pool.reserve(static_cast<size_t>(4 * topk));
    for (int i = 0; i < n; ++i) {
        if (mark[static_cast<size_t>(i)]) {
            pool.push_back(i);
        }
    }
    if (pool.empty()) {
        pool = idx;
    }
    return pool;
}

double centrality_key(const TreePose& p) {
    return std::max(std::abs(p.x), std::abs(p.y));
}

std::pair<int, int> find_first_overlap(const std::vector<Point>& centers,
                                       const std::vector<BoundingBox>& bbs,
                                       const std::vector<Polygon>& polys,
                                       double limit_sq) {
    const int n = static_cast<int>(centers.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = centers[static_cast<size_t>(i)].x - centers[static_cast<size_t>(j)].x;
            const double dy = centers[static_cast<size_t>(i)].y - centers[static_cast<size_t>(j)].y;
            if (dx * dx + dy * dy > limit_sq) {
                continue;
            }
            if (!aabb_overlap(bbs[static_cast<size_t>(i)], bbs[static_cast<size_t>(j)])) {
                continue;
            }
            if (polygons_intersect(polys[static_cast<size_t>(i)], polys[static_cast<size_t>(j)])) {
                return {i, j};
            }
        }
    }
    return {-1, -1};
}

bool try_relocate_one(const Polygon& base_poly,
                      double limit_sq,
                      int idx,
                      int other,
                      std::vector<TreePose>& poses,
                      std::vector<Point>& centers,
                      std::vector<Polygon>& polys,
                      std::vector<BoundingBox>& bbs,
                      std::mt19937_64& rng,
                      const Options& opt) {
    const int n = static_cast<int>(poses.size());
    if (idx < 0 || idx >= n) {
        return false;
    }

    const Extents e_without = compute_extents_excluding(bbs, idx);
    const Extents e_all = compute_extents(bbs);
    const double curr_side = side_from_extents(e_all);
    const double base_step =
        std::max(1e-6, opt.repair_step0_frac * (curr_side > 0.0 ? curr_side : 1.0));

    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::normal_distribution<double> normal01(0.0, 1.0);
    std::uniform_real_distribution<double> uni_deg(-opt.repair_ddeg_max, opt.repair_ddeg_max);

    const Point c0 = centers[static_cast<size_t>(idx)];
    const Point c_other = centers[static_cast<size_t>(other)];
    double dir_x = c0.x - c_other.x;
    double dir_y = c0.y - c_other.y;
    double dir_norm = std::hypot(dir_x, dir_y);
    if (!(dir_norm > 1e-12)) {
        std::uniform_real_distribution<double> ang(0.0, 2.0 * 3.14159265358979323846);
        double a = ang(rng);
        dir_x = std::cos(a);
        dir_y = std::sin(a);
        dir_norm = 1.0;
    }
    dir_x /= dir_norm;
    dir_y /= dir_norm;

    bool found = false;
    double best_side = std::numeric_limits<double>::infinity();
    TreePose best_pose = poses[static_cast<size_t>(idx)];
    Polygon best_poly;
    BoundingBox best_bb{};

    for (int attempt = 0; attempt < opt.repair_attempts; ++attempt) {
        const double step = base_step * std::pow(opt.repair_step_mult, attempt);
        const double noise = opt.repair_noise_frac * (curr_side > 0.0 ? curr_side : 1.0);
        const double margin = 0.02 * (curr_side > 0.0 ? curr_side : 1.0) + step;

        TreePose cand = poses[static_cast<size_t>(idx)];
        if (uni01(rng) < opt.repair_p_uniform) {
            std::uniform_real_distribution<double> ux(e_without.min_x - margin, e_without.max_x + margin);
            std::uniform_real_distribution<double> uy(e_without.min_y - margin, e_without.max_y + margin);
            cand.x = ux(rng);
            cand.y = uy(rng);
        } else {
            cand.x = c0.x + dir_x * step + normal01(rng) * noise;
            cand.y = c0.y + dir_y * step + normal01(rng) * noise;
        }

        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
            continue;
        }

        if (uni01(rng) < opt.repair_p_rot) {
            cand.deg = wrap_deg(cand.deg + uni_deg(rng));
        } else {
            cand.deg = wrap_deg(cand.deg);
        }

        Polygon cand_poly = transform_polygon(base_poly, cand);
        BoundingBox cand_bb = bounding_box(cand_poly);

        bool collide = false;
        for (int j = 0; j < n; ++j) {
            if (j == idx) {
                continue;
            }
            const double dx = cand.x - centers[static_cast<size_t>(j)].x;
            const double dy = cand.y - centers[static_cast<size_t>(j)].y;
            if (dx * dx + dy * dy > limit_sq) {
                continue;
            }
            if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                continue;
            }
            if (polygons_intersect(cand_poly, polys[static_cast<size_t>(j)])) {
                collide = true;
                break;
            }
        }
        if (collide) {
            continue;
        }

        Extents e_new = merge_extents_bb(e_without, cand_bb);
        double side_new = side_from_extents(e_new);
        if (!found || side_new + 1e-15 < best_side) {
            found = true;
            best_side = side_new;
            best_pose = cand;
            best_poly = std::move(cand_poly);
            best_bb = cand_bb;
        }
    }

    if (!found) {
        return false;
    }

    poses[static_cast<size_t>(idx)] = best_pose;
    centers[static_cast<size_t>(idx)] = Point{best_pose.x, best_pose.y};
    polys[static_cast<size_t>(idx)] = std::move(best_poly);
    bbs[static_cast<size_t>(idx)] = best_bb;
    return true;
}

bool greedy_rebuild(const Polygon& base_poly,
                    double radius,
                    std::vector<TreePose>& poses,
                    std::mt19937_64& rng,
                    const Options& opt) {
    const int n = static_cast<int>(poses.size());
    if (n <= 0) {
        return true;
    }

    std::vector<int> order(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return centrality_key(poses[static_cast<size_t>(a)]) <
               centrality_key(poses[static_cast<size_t>(b)]);
    });

    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::normal_distribution<double> normal01(0.0, 1.0);
    std::uniform_real_distribution<double> uni_deg(-opt.repair_ddeg_max, opt.repair_ddeg_max);
    std::uniform_real_distribution<double> uni_ang(0.0, 2.0 * 3.14159265358979323846);

    const double thr = 2.0 * radius + 1e-9;
    const double limit_sq = thr * thr;

    std::vector<TreePose> placed;
    placed.reserve(static_cast<size_t>(n));
    std::vector<Point> centers;
    centers.reserve(static_cast<size_t>(n));
    std::vector<Polygon> polys;
    polys.reserve(static_cast<size_t>(n));
    std::vector<BoundingBox> bbs;
    bbs.reserve(static_cast<size_t>(n));
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();

    for (int t = 0; t < n; ++t) {
        const TreePose base = poses[static_cast<size_t>(order[static_cast<size_t>(t)])];
        const double curr_side =
            (t == 0) ? 0.0 : std::max(e.max_x - e.min_x, e.max_y - e.min_y);
        const double scale = std::max(2.0 * radius, curr_side > 0.0 ? curr_side : 1.0);
        const double step0 = std::max(1e-6, opt.repair_step0_frac * scale);
        const double noise = opt.repair_noise_frac * scale;

        bool found = false;
        double best_side = std::numeric_limits<double>::infinity();
        TreePose best_pose = base;
        Polygon best_poly;
        BoundingBox best_bb{};

        for (int attempt = 0; attempt < opt.repair_attempts; ++attempt) {
            TreePose cand = base;
            if (attempt > 0) {
                const double step = step0 * std::pow(opt.repair_step_mult, attempt - 1);
                if (uni01(rng) < opt.repair_p_uniform && t > 0) {
                    const double margin = 0.03 * scale + step;
                    std::uniform_real_distribution<double> ux(std::min(e.min_x, base.x) - margin,
                                                              std::max(e.max_x, base.x) + margin);
                    std::uniform_real_distribution<double> uy(std::min(e.min_y, base.y) - margin,
                                                              std::max(e.max_y, base.y) + margin);
                    cand.x = ux(rng);
                    cand.y = uy(rng);
                } else {
                    const double a = uni_ang(rng);
                    cand.x = base.x + std::cos(a) * step + normal01(rng) * noise;
                    cand.y = base.y + std::sin(a) * step + normal01(rng) * noise;
                }

                if (uni01(rng) < opt.repair_p_rot) {
                    cand.deg = wrap_deg(cand.deg + uni_deg(rng));
                } else {
                    cand.deg = wrap_deg(cand.deg);
                }
            } else {
                cand.deg = wrap_deg(cand.deg);
            }

            if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
                continue;
            }

            Polygon cand_poly = transform_polygon(base_poly, cand);
            BoundingBox cand_bb = bounding_box(cand_poly);

            bool collide = false;
            for (size_t j = 0; j < placed.size(); ++j) {
                const double dx = cand.x - centers[j].x;
                const double dy = cand.y - centers[j].y;
                if (dx * dx + dy * dy > limit_sq) {
                    continue;
                }
                if (!aabb_overlap(cand_bb, bbs[j])) {
                    continue;
                }
                if (polygons_intersect(cand_poly, polys[j])) {
                    collide = true;
                    break;
                }
            }
            if (collide) {
                continue;
            }

            Extents e_new = e;
            if (t == 0) {
                e_new.min_x = cand_bb.min_x;
                e_new.max_x = cand_bb.max_x;
                e_new.min_y = cand_bb.min_y;
                e_new.max_y = cand_bb.max_y;
            } else {
                e_new = merge_extents_bb(e, cand_bb);
            }
            double side_new = side_from_extents(e_new);
            if (!found || side_new + 1e-15 < best_side) {
                found = true;
                best_side = side_new;
                best_pose = cand;
                best_poly = std::move(cand_poly);
                best_bb = cand_bb;
            }
        }

        if (!found) {
            return false;
        }

        placed.push_back(best_pose);
        centers.push_back(Point{best_pose.x, best_pose.y});
        polys.push_back(std::move(best_poly));
        bbs.push_back(best_bb);
        if (t == 0) {
            e.min_x = best_bb.min_x;
            e.max_x = best_bb.max_x;
            e.min_y = best_bb.min_y;
            e.max_y = best_bb.max_y;
        } else {
            e = merge_extents_bb(e, best_bb);
        }
    }

    poses = std::move(placed);
    return true;
}

bool repair_inplace(const Polygon& base_poly,
                    double radius,
                    std::vector<TreePose>& poses,
                    const std::vector<char>& movable,
                    uint64_t seed,
                    const Options& opt) {
    const int n = static_cast<int>(poses.size());
    if (n <= 1) {
        return true;
    }

    const double thr = 2.0 * radius + 1e-9;
    const double limit_sq = thr * thr;

    std::vector<Point> centers;
    centers.reserve(poses.size());
    std::vector<Polygon> polys;
    polys.reserve(poses.size());
    std::vector<BoundingBox> bbs;
    bbs.reserve(poses.size());
    for (const auto& p : poses) {
        centers.push_back(Point{p.x, p.y});
        Polygon poly = transform_polygon(base_poly, p);
        polys.push_back(std::move(poly));
        bbs.push_back(bounding_box(polys.back()));
    }

    std::mt19937_64 rng(seed);
    for (int pass = 0; pass < opt.repair_passes; ++pass) {
        auto [i, j] = find_first_overlap(centers, bbs, polys, limit_sq);
        if (i < 0) {
            return true;
        }

        auto pick_idx = [&](int a, int b) -> int {
            const bool ma = (a >= 0 && a < static_cast<int>(movable.size())) ? (movable[static_cast<size_t>(a)] != 0) : true;
            const bool mb = (b >= 0 && b < static_cast<int>(movable.size())) ? (movable[static_cast<size_t>(b)] != 0) : true;
            if (ma != mb) {
                return ma ? a : b;
            }
            double ka = centrality_key(poses[static_cast<size_t>(a)]);
            double kb = centrality_key(poses[static_cast<size_t>(b)]);
            return (ka < kb) ? a : b;
        };

        int idx = pick_idx(i, j);
        int other = (idx == i) ? j : i;
        if (try_relocate_one(base_poly, limit_sq, idx, other, poses, centers, polys, bbs, rng, opt)) {
            continue;
        }
        int idx2 = other;
        int other2 = idx;
        if (try_relocate_one(base_poly, limit_sq, idx2, other2, poses, centers, polys, bbs, rng, opt)) {
            continue;
        }

        if (!greedy_rebuild(base_poly, radius, poses, rng, opt)) {
            return false;
        }

        centers.clear();
        polys.clear();
        bbs.clear();
        centers.reserve(poses.size());
        polys.reserve(poses.size());
        bbs.reserve(poses.size());
        for (const auto& p : poses) {
            centers.push_back(Point{p.x, p.y});
            Polygon poly = transform_polygon(base_poly, p);
            polys.push_back(std::move(poly));
            bbs.push_back(bounding_box(polys.back()));
        }

        if (find_first_overlap(centers, bbs, polys, limit_sq).first < 0) {
            return true;
        }
    }

    return find_first_overlap(centers, bbs, polys, limit_sq).first < 0;
}

bool make_candidate(const Polygon& base_poly,
                    double radius,
                    const std::vector<TreePose>& poses_in,
                    const Options& opt,
                    CandidateSol& out) {
    const int n = static_cast<int>(poses_in.size());
    if (n <= 0) {
        return false;
    }

    std::vector<TreePose> poses = quantize_poses(poses_in, opt.output_decimals);
    for (const auto& p : poses) {
        if (p.x < -100.0 || p.x > 100.0 || p.y < -100.0 || p.y > 100.0) {
            return false;
        }
    }

    if (any_overlap(base_poly, poses, radius)) {
        return false;
    }

    CandidateSol sol;
    sol.poses = std::move(poses);
    sol.bbs = bounding_boxes_for_poses(base_poly, sol.poses);
    sol.side = side_from_extents(compute_extents(sol.bbs));
    sol.boundary_pool = build_extreme_pool(sol.bbs, std::max(1, opt.boundary_topk));
    out = std::move(sol);
    return true;
}

bool finalize_solution(const Polygon& base_poly,
                       double radius,
                       std::vector<TreePose>& poses,
                       uint64_t seed,
                       const Options& opt,
                       CandidateSol& out) {
    if (poses.empty()) {
        return false;
    }

    if (opt.final_rigid) {
        optimize_rigid_rotation(base_poly, poses);
    }

    CandidateSol sol;
    if (make_candidate(base_poly, radius, poses, opt, sol)) {
        out = std::move(sol);
        return true;
    }

    std::vector<TreePose> q = quantize_poses(poses, opt.output_decimals);
    std::vector<char> all_movable(q.size(), 1);
    Options opt2 = opt;
    opt2.repair_passes = std::max(20, opt.repair_passes / 4);
    opt2.repair_attempts = std::max(20, opt.repair_attempts / 2);
    if (!repair_inplace(base_poly, radius, q, all_movable, seed ^ 0xD1B54A32D192ED03ULL, opt2)) {
        return false;
    }
    if (opt.final_rigid) {
        optimize_rigid_rotation(base_poly, q);
    }
    if (!make_candidate(base_poly, radius, q, opt, sol)) {
        return false;
    }
    out = std::move(sol);
    return true;
}

int pick_ranked(int k, std::mt19937_64& rng) {
    if (k <= 1) {
        return 0;
    }
    std::vector<double> w(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        w[static_cast<size_t>(i)] = 1.0 / static_cast<double>(i + 1);
    }
    std::discrete_distribution<int> dist(w.begin(), w.end());
    return dist(rng);
}

Options parse_args(int argc, char** argv, std::vector<std::string>& inputs) {
    if (argc < 3) {
        throw std::runtime_error("Uso: blend_repair output.csv input1.csv [input2.csv ...] [--opções]");
    }

    Options opt;

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
    auto parse_double = [](const std::string& s) -> double {
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Double inválido: " + s);
        }
        return v;
    };

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Faltou valor para " + name);
            }
            return argv[++i];
        };

        if (arg == "--n-min") {
            opt.n_min = parse_int(need(arg));
        } else if (arg == "--n-max") {
            opt.n_max = parse_int(need(arg));
        } else if (arg == "--seed") {
            opt.seed = parse_u64(need(arg));
        } else if (arg == "--topk-per-n") {
            opt.topk_per_n = parse_int(need(arg));
        } else if (arg == "--blend-iters") {
            opt.blend_iters = parse_int(need(arg));
        } else if (arg == "--boundary-topk") {
            opt.boundary_topk = parse_int(need(arg));
        } else if (arg == "--replace-min") {
            opt.replace_min = parse_int(need(arg));
        } else if (arg == "--replace-max") {
            opt.replace_max = parse_int(need(arg));
        } else if (arg == "--repair-passes") {
            opt.repair_passes = parse_int(need(arg));
        } else if (arg == "--repair-attempts") {
            opt.repair_attempts = parse_int(need(arg));
        } else if (arg == "--repair-step0-frac") {
            opt.repair_step0_frac = parse_double(need(arg));
        } else if (arg == "--repair-step-mult") {
            opt.repair_step_mult = parse_double(need(arg));
        } else if (arg == "--repair-noise-frac") {
            opt.repair_noise_frac = parse_double(need(arg));
        } else if (arg == "--repair-ddeg-max") {
            opt.repair_ddeg_max = parse_double(need(arg));
        } else if (arg == "--repair-p-uniform") {
            opt.repair_p_uniform = parse_double(need(arg));
        } else if (arg == "--repair-p-rot") {
            opt.repair_p_rot = parse_double(need(arg));
        } else if (arg == "--sa-restarts") {
            opt.sa_restarts = parse_int(need(arg));
        } else if (arg == "--sa-iters") {
            opt.sa_iters = parse_int(need(arg));
        } else if (arg == "--sa-w-micro") {
            opt.sa_w_micro = parse_double(need(arg));
        } else if (arg == "--sa-w-swap-rot") {
            opt.sa_w_swap_rot = parse_double(need(arg));
        } else if (arg == "--sa-w-relocate") {
            opt.sa_w_relocate = parse_double(need(arg));
        } else if (arg == "--sa-w-block-translate") {
            opt.sa_w_block_translate = parse_double(need(arg));
        } else if (arg == "--sa-w-block-rotate") {
            opt.sa_w_block_rotate = parse_double(need(arg));
        } else if (arg == "--sa-w-lns") {
            opt.sa_w_lns = parse_double(need(arg));
        } else if (arg == "--sa-block-size") {
            opt.sa_block_size = parse_int(need(arg));
        } else if (arg == "--sa-lns-remove") {
            opt.sa_lns_remove = parse_int(need(arg));
        } else if (arg == "--sa-hh-segment") {
            opt.sa_hh_segment = parse_int(need(arg));
        } else if (arg == "--sa-hh-reaction") {
            opt.sa_hh_reaction = parse_double(need(arg));
        } else if (arg == "--squeeze-tries") {
            opt.squeeze_tries = parse_int(need(arg));
        } else if (arg == "--squeeze-alpha-min") {
            opt.squeeze_alpha_min = parse_double(need(arg));
        } else if (arg == "--squeeze-alpha-max") {
            opt.squeeze_alpha_max = parse_double(need(arg));
        } else if (arg == "--squeeze-p-aniso") {
            opt.squeeze_p_aniso = parse_double(need(arg));
        } else if (arg == "--squeeze-repair-passes") {
            opt.squeeze_repair_passes = parse_int(need(arg));
        } else if (arg == "--no-final-rigid") {
            opt.final_rigid = false;
        } else if (arg == "--output-decimals") {
            opt.output_decimals = parse_int(need(arg));
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("Argumento desconhecido: " + arg);
        } else {
            inputs.push_back(std::move(arg));
        }
    }

    if (inputs.empty()) {
        throw std::runtime_error("Precisa de ao menos 1 input.");
    }
    if (opt.n_min < 1 || opt.n_min > 200 || opt.n_max < 1 || opt.n_max > 200 || opt.n_min > opt.n_max) {
        throw std::runtime_error("--n-min/--n-max inválidos.");
    }
    if (opt.topk_per_n <= 0) {
        throw std::runtime_error("--topk-per-n precisa ser > 0.");
    }
    if (opt.blend_iters < 0) {
        throw std::runtime_error("--blend-iters precisa ser >= 0.");
    }
    if (opt.boundary_topk <= 0) {
        throw std::runtime_error("--boundary-topk precisa ser > 0.");
    }
    if (opt.replace_min < 0 || opt.replace_max < 0 || opt.replace_min > opt.replace_max) {
        throw std::runtime_error("--replace-min/--replace-max inválidos.");
    }
    if (opt.repair_passes < 0 || opt.repair_attempts <= 0) {
        throw std::runtime_error("--repair-passes precisa ser >= 0 e --repair-attempts > 0.");
    }
    if (!(opt.repair_step0_frac >= 0.0) || !(opt.repair_step_mult >= 1.0) ||
        !(opt.repair_noise_frac >= 0.0) || !(opt.repair_ddeg_max >= 0.0)) {
        throw std::runtime_error("Parâmetros de repair inválidos.");
    }
    if (opt.repair_p_uniform < 0.0 || opt.repair_p_uniform > 1.0 ||
        opt.repair_p_rot < 0.0 || opt.repair_p_rot > 1.0) {
        throw std::runtime_error("--repair-p-uniform/--repair-p-rot precisam estar em [0,1].");
    }
    if (opt.output_decimals < 0 || opt.output_decimals > 18) {
        throw std::runtime_error("--output-decimals precisa estar em [0,18].");
    }
    if (opt.sa_restarts < 0 || opt.sa_iters < 0) {
        throw std::runtime_error("--sa-restarts/--sa-iters precisam ser >= 0.");
    }
    if (opt.sa_w_micro < 0.0 || opt.sa_w_swap_rot < 0.0 || opt.sa_w_relocate < 0.0 ||
        opt.sa_w_block_translate < 0.0 || opt.sa_w_block_rotate < 0.0 || opt.sa_w_lns < 0.0) {
        throw std::runtime_error("Pesos de SA precisam ser >= 0.");
    }
    if (opt.sa_block_size <= 0) {
        throw std::runtime_error("--sa-block-size precisa ser > 0.");
    }
    if (opt.sa_lns_remove < 0) {
        throw std::runtime_error("--sa-lns-remove precisa ser >= 0.");
    }
    if (opt.sa_hh_segment < 0) {
        throw std::runtime_error("--sa-hh-segment precisa ser >= 0.");
    }
    if (opt.sa_hh_reaction < 0.0 || opt.sa_hh_reaction > 1.0) {
        throw std::runtime_error("--sa-hh-reaction precisa estar em [0,1].");
    }
    if (opt.squeeze_tries < 0) {
        throw std::runtime_error("--squeeze-tries precisa ser >= 0.");
    }
    if (!(opt.squeeze_alpha_min > 0.0) || !(opt.squeeze_alpha_max > 0.0) ||
        opt.squeeze_alpha_min > opt.squeeze_alpha_max || opt.squeeze_alpha_max > 1.0) {
        throw std::runtime_error("--squeeze-alpha-min/max inválidos (0 < min <= max <= 1).");
    }
    if (opt.squeeze_p_aniso < 0.0 || opt.squeeze_p_aniso > 1.0) {
        throw std::runtime_error("--squeeze-p-aniso precisa estar em [0,1].");
    }
    if (opt.squeeze_repair_passes < 0) {
        throw std::runtime_error("--squeeze-repair-passes precisa ser >= 0.");
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Uso: " << argv[0]
                << " output.csv input1.csv [input2.csv ...] [--opções]\n";
            return 2;
        }

        const std::string output_path = argv[1];
        std::vector<std::string> inputs;
        Options opt = parse_args(argc, argv, inputs);

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);

        std::vector<SubmissionData> subs;
        subs.reserve(inputs.size());
        for (const auto& p : inputs) {
            subs.push_back(load_submission(p));
        }

        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("Erro ao abrir arquivo de saída: " + output_path);
        }
        out << "id,x,y,deg\n";

        SARefiner sa(base_poly, radius);

        double total_score = 0.0;
        int improved_n = 0;

        for (int n = opt.n_min; n <= opt.n_max; ++n) {
            std::vector<CandidateSol> pool;
            pool.reserve(subs.size());

            for (int si = 0; si < static_cast<int>(subs.size()); ++si) {
                const auto& sub = subs[static_cast<size_t>(si)];
                const auto& recs = sub.by_n[static_cast<size_t>(n)];
                std::vector<TreePose> poses;
                poses.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    const auto& r = recs[static_cast<size_t>(i)];
                    poses.push_back(TreePose{r.x, r.y, r.deg});
                }

                CandidateSol sol;
                std::vector<TreePose> tmp = std::move(poses);
                uint64_t seed =
                    opt.seed ^
                    (0x9e3779b97f4a7c15ULL +
                     static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
                     static_cast<uint64_t>(si) * 0x94d049bb133111ebULL);
                if (!finalize_solution(base_poly, radius, tmp, seed, opt, sol)) {
                    continue;
                }
                pool.push_back(std::move(sol));
            }

            if (pool.empty()) {
                throw std::runtime_error("Nenhum input válido para n=" + std::to_string(n));
            }

            std::sort(pool.begin(), pool.end(), [](const CandidateSol& a, const CandidateSol& b) {
                return a.side < b.side;
            });
            if (static_cast<int>(pool.size()) > opt.topk_per_n) {
                pool.resize(static_cast<size_t>(opt.topk_per_n));
            }

            const double best_parent_side = pool[0].side;
            CandidateSol best = pool[0];

            if (opt.blend_iters > 0 && pool.size() >= 2 && opt.repair_passes > 0) {
                std::mt19937_64 rng(opt.seed ^
                                    (0x9e3779b97f4a7c15ULL +
                                     static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL));
                std::uniform_int_distribution<int> uni_replace(0, 1);

                for (int it = 0; it < opt.blend_iters; ++it) {
                    int a = pick_ranked(static_cast<int>(pool.size()), rng);
                    int b = pick_ranked(static_cast<int>(pool.size()), rng);
                    if (b == a && pool.size() >= 2) {
                        b = (b + 1) % static_cast<int>(pool.size());
                    }

                    const auto& pa = pool[static_cast<size_t>(a)];
                    const auto& pb = pool[static_cast<size_t>(b)];

                    const int nposes = static_cast<int>(pa.poses.size());
                    if (nposes != n) {
                        continue;
                    }

                    const auto& pool_a = pa.boundary_pool;
                    const auto& pool_b = pb.boundary_pool;
                    if (pool_a.empty() || pool_b.empty()) {
                        continue;
                    }

                    const int max_replace = std::min({opt.replace_max,
                                                     static_cast<int>(pool_a.size()),
                                                     static_cast<int>(pool_b.size()),
                                                     n});
                    int min_replace = std::min(opt.replace_min, max_replace);
                    if (max_replace <= 0 || min_replace < 0) {
                        continue;
                    }
                    if (min_replace == 0 && max_replace > 0 && uni_replace(rng) == 0) {
                        continue;
                    }
                    if (min_replace <= 0) {
                        min_replace = 1;
                    }

                    std::uniform_int_distribution<int> uni_k(min_replace, max_replace);
                    const int k = uni_k(rng);

                    std::vector<int> idx_a = pool_a;
                    std::vector<int> idx_b = pool_b;
                    std::shuffle(idx_a.begin(), idx_a.end(), rng);
                    std::shuffle(idx_b.begin(), idx_b.end(), rng);

                    std::vector<TreePose> child = pa.poses;
                    std::vector<char> movable(static_cast<size_t>(n), 0);
                    for (int t = 0; t < k; ++t) {
                        int ia = idx_a[static_cast<size_t>(t)];
                        int ib = idx_b[static_cast<size_t>(t)];
                        if (ia < 0 || ia >= n) {
                            continue;
                        }
                        child[static_cast<size_t>(ia)] = pb.poses[static_cast<size_t>(ib)];
                        movable[static_cast<size_t>(ia)] = 1;
                    }

                    if (!repair_inplace(base_poly,
                                        radius,
                                        child,
                                        movable,
                                        opt.seed ^
                                            (0x94d049bb133111ebULL +
                                             static_cast<uint64_t>(n) * 0x2545F4914F6CDD1DULL +
                                             static_cast<uint64_t>(it) * 0xD6E8FEB86659FD93ULL),
                                        opt)) {
                        continue;
                    }

                    std::vector<TreePose> refined = std::move(child);
                    if (opt.sa_iters > 0 && opt.sa_restarts > 0) {
                        SARefiner::Params p;
                        p.iters = opt.sa_iters;
                        p.w_micro = opt.sa_w_micro;
                        p.w_swap_rot = opt.sa_w_swap_rot;
                        p.w_relocate = opt.sa_w_relocate;
                        p.w_block_translate = opt.sa_w_block_translate;
                        p.w_block_rotate = opt.sa_w_block_rotate;
                        p.w_lns = opt.sa_w_lns;
                        p.block_size = opt.sa_block_size;
                        p.lns_remove = opt.sa_lns_remove;
                        p.hh_segment = opt.sa_hh_segment;
                        p.hh_reaction = opt.sa_hh_reaction;

                        double best_sa_side = std::numeric_limits<double>::infinity();
                        std::vector<TreePose> best_sa = refined;
                        for (int r = 0; r < opt.sa_restarts; ++r) {
                            uint64_t s =
                                opt.seed ^
                                (0x9e3779b97f4a7c15ULL +
                                 static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
                                 static_cast<uint64_t>(it) * 0x94d049bb133111ebULL +
                                 static_cast<uint64_t>(r) * 0x2545F4914F6CDD1DULL);
                            SARefiner::Result res = sa.refine_min_side(refined, s, p);
                            CandidateSol cand;
                            std::vector<TreePose> tmp = std::move(res.best_poses);
                            if (!finalize_solution(base_poly, radius, tmp, s, opt, cand)) {
                                continue;
                            }
                            if (cand.side + 1e-15 < best_sa_side) {
                                best_sa_side = cand.side;
                                best_sa = std::move(cand.poses);
                            }
                        }
                        refined = std::move(best_sa);
                    }

                    CandidateSol child_sol;
                    {
                        std::vector<TreePose> tmp = std::move(refined);
                        if (!finalize_solution(base_poly,
                                               radius,
                                               tmp,
                                               opt.seed ^
                                                   (0x9e3779b97f4a7c15ULL +
                                                    static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
                                                    static_cast<uint64_t>(it) * 0x94d049bb133111ebULL),
                                               opt,
                                               child_sol)) {
                            continue;
                        }
                    }

                    if (child_sol.side + 1e-15 < best.side) {
                        best = child_sol;
                    }

                    if (child_sol.side + 1e-15 < pool.back().side) {
                        auto it_pos = std::lower_bound(pool.begin(),
                                                       pool.end(),
                                                       child_sol.side,
                                                       [](const CandidateSol& s, double v) {
                                                           return s.side < v;
                                                       });
                        pool.insert(it_pos, child_sol);
                        if (static_cast<int>(pool.size()) > opt.topk_per_n) {
                            pool.resize(static_cast<size_t>(opt.topk_per_n));
                        }
                    }
                }
            }

            if (opt.squeeze_tries > 0 && opt.squeeze_repair_passes > 0 && opt.repair_attempts > 0) {
                Options opt_sq = opt;
                opt_sq.repair_passes = opt.squeeze_repair_passes;

                std::mt19937_64 rng(opt.seed ^
                                    (0x9e3779b97f4a7c15ULL +
                                     static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
                                     0xD1B54A32D192ED03ULL));
                std::uniform_real_distribution<double> uni01(0.0, 1.0);
                std::uniform_real_distribution<double> uni_alpha(opt.squeeze_alpha_min,
                                                                 opt.squeeze_alpha_max);

                for (int st = 0; st < opt.squeeze_tries; ++st) {
                    const Extents e = compute_extents(best.bbs);
                    const double cx = 0.5 * (e.min_x + e.max_x);
                    const double cy = 0.5 * (e.min_y + e.max_y);

	                    double ax = 1.0;
	                    double ay = 1.0;
	                    if (uni01(rng) < opt.squeeze_p_aniso) {
	                        const double w = e.max_x - e.min_x;
	                        const double h = e.max_y - e.min_y;
	                        const bool prefer_x = (w >= h);
	                        bool shrink_x = prefer_x;
	                        if (uni01(rng) < 0.25) {
	                            shrink_x = !shrink_x;
	                        }
	                        if (shrink_x) {
	                            ax = uni_alpha(rng);
	                        } else {
	                            ay = uni_alpha(rng);
	                        }
	                    } else {
	                        ax = uni_alpha(rng);
	                        ay = uni_alpha(rng);
	                    }
                    if (!(ax < 1.0) && !(ay < 1.0)) {
                        continue;
                    }

                    std::vector<TreePose> tmp = best.poses;
                    for (auto& p : tmp) {
                        p.x = cx + ax * (p.x - cx);
                        p.y = cy + ay * (p.y - cy);
                    }

                    std::vector<char> movable(tmp.size(), 1);
                    uint64_t s = opt.seed ^
                                 (0x94d049bb133111ebULL +
                                  static_cast<uint64_t>(n) * 0x2545F4914F6CDD1DULL +
                                  static_cast<uint64_t>(st) * 0xBF58476D1CE4E5B9ULL);
	                    if (!repair_inplace(base_poly, radius, tmp, movable, s, opt_sq)) {
	                        continue;
	                    }

	                    if (opt.sa_iters > 0 && opt.sa_restarts > 0) {
	                        SARefiner::Params p;
	                        p.iters = opt.sa_iters;
	                        p.w_micro = opt.sa_w_micro;
	                        p.w_swap_rot = opt.sa_w_swap_rot;
	                        p.w_relocate = opt.sa_w_relocate;
	                        p.w_block_translate = opt.sa_w_block_translate;
	                        p.w_block_rotate = opt.sa_w_block_rotate;
	                        p.w_lns = opt.sa_w_lns;
	                        p.block_size = opt.sa_block_size;
	                        p.lns_remove = opt.sa_lns_remove;
	                        p.hh_segment = opt.sa_hh_segment;
	                        p.hh_reaction = opt.sa_hh_reaction;

	                        double best_sa_side = std::numeric_limits<double>::infinity();
	                        std::vector<TreePose> best_sa = tmp;
	                        for (int r = 0; r < opt.sa_restarts; ++r) {
	                            uint64_t sr =
	                                s ^
	                                (0x9e3779b97f4a7c15ULL +
	                                 static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
	                                 static_cast<uint64_t>(r) * 0x94d049bb133111ebULL);
	                            SARefiner::Result res = sa.refine_min_side(tmp, sr, p);
	                            CandidateSol cand;
	                            std::vector<TreePose> tmp2 = std::move(res.best_poses);
	                            if (!finalize_solution(base_poly, radius, tmp2, sr, opt, cand)) {
	                                continue;
	                            }
	                            if (cand.side + 1e-15 < best_sa_side) {
	                                best_sa_side = cand.side;
	                                best_sa = std::move(cand.poses);
	                            }
	                        }
	                        tmp = std::move(best_sa);
	                    }

	                    CandidateSol cand;
	                    if (!finalize_solution(base_poly, radius, tmp, s, opt, cand)) {
	                        continue;
	                    }
                    if (cand.side + 1e-15 < best.side) {
                        best = std::move(cand);
                    }
                }
            }

            if (best.side + 1e-15 < best_parent_side) {
                improved_n += 1;
            }

            total_score += (best.side * best.side) / static_cast<double>(n);

            for (int i = 0; i < n; ++i) {
                const auto& p = best.poses[static_cast<size_t>(i)];
                out << std::setw(3) << std::setfill('0') << n << "_" << i << ","
                    << fmt_submission_value(p.x, opt.output_decimals) << ","
                    << fmt_submission_value(p.y, opt.output_decimals) << ","
                    << fmt_submission_value(p.deg, opt.output_decimals) << "\n";
            }
        }

        std::cout << "Submission blend+repair gerada em " << output_path << "\n";
        std::cout << "Inputs: " << inputs.size() << "\n";
        std::cout << "n range: [" << opt.n_min << ", " << opt.n_max << "]\n";
        std::cout << "TopK per n: " << opt.topk_per_n << "\n";
        std::cout << "Blend iters: " << opt.blend_iters << "\n";
        std::cout << "Repair passes: " << opt.repair_passes << "\n";
        std::cout << "Squeeze: " << (opt.squeeze_tries > 0 && opt.squeeze_repair_passes > 0 ? "on" : "off")
                  << " (tries=" << opt.squeeze_tries
                  << ", alpha=[" << opt.squeeze_alpha_min << "," << opt.squeeze_alpha_max << "]"
                  << ", aniso=" << opt.squeeze_p_aniso
                  << ", repair_passes=" << opt.squeeze_repair_passes << ")\n";
        std::cout << "SA: " << (opt.sa_iters > 0 && opt.sa_restarts > 0 ? "on" : "off")
                  << " (iters=" << opt.sa_iters << ", restarts=" << opt.sa_restarts << ")\n";
        std::cout << "Final rigid: " << (opt.final_rigid ? "on" : "off") << "\n";
        std::cout << "Output decimals: " << opt.output_decimals << "\n";
        std::cout << "Improved n vs best parent: " << improved_n << "\n";
        std::cout << "Score (local): " << std::fixed << std::setprecision(9)
                  << total_score << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
