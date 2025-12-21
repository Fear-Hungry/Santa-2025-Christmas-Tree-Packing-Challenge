
	                if (n <= 2 || p.lns_remove <= 0) {
	                    add_reward(op, false, false, false, curr_cost, curr_cost);
	                    maybe_update_controller(t);
	                    continue;
	                }

                const double boundary_tol = 1e-9;
                std::vector<int> boundary;
                boundary.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    if (!is_active(i)) {
                        continue;
                    }
                    const auto& bb = bbs[static_cast<size_t>(i)];
                    if (bb.min_x <= gmnx + boundary_tol || bb.max_x >= gmxx - boundary_tol ||
                        bb.min_y <= gmny + boundary_tol || bb.max_y >= gmxy - boundary_tol) {
                        boundary.push_back(i);
                    }
                }
                if (boundary.empty()) {
                    boundary = picker.boundary_pool();
                }
                if (boundary.empty()) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                int m = std::min(std::max(1, p.lns_remove), n - 1);
                if (static_cast<int>(boundary.size()) < m) {
                    m = static_cast<int>(boundary.size());
                }

                const double tol = 1e-9;
                auto two_smallest = [&](auto get) -> std::tuple<double, double, int> {
                    double m1 = std::numeric_limits<double>::infinity();
                    double m2 = std::numeric_limits<double>::infinity();
                    int c1 = 0;
                    for (const auto& bb : bbs) {
                        double v = get(bb);
                        if (v < m1 - tol) {
                            m2 = m1;
                            m1 = v;
                            c1 = 1;
                        } else if (std::abs(v - m1) <= tol) {
                            c1 += 1;
                        } else if (v < m2 - tol) {
                            m2 = v;
                        }
                    }
                    if (!std::isfinite(m2)) {
                        m2 = m1;
                    }
                    return {m1, m2, c1};
                };
                auto two_largest = [&](auto get) -> std::tuple<double, double, int> {
                    double m1 = -std::numeric_limits<double>::infinity();
                    double m2 = -std::numeric_limits<double>::infinity();
                    int c1 = 0;
                    for (const auto& bb : bbs) {
                        double v = get(bb);
                        if (v > m1 + tol) {
                            m2 = m1;
                            m1 = v;
                            c1 = 1;
                        } else if (std::abs(v - m1) <= tol) {
                            c1 += 1;
                        } else if (v > m2 + tol) {
                            m2 = v;
                        }
                    }
                    if (!std::isfinite(m2)) {
                        m2 = m1;
                    }
                    return {m1, m2, c1};
                };

                auto [minx1, minx2, minx_c] =
                    two_smallest([&](const BoundingBox& bb) { return bb.min_x; });
                auto [maxx1, maxx2, maxx_c] =
                    two_largest([&](const BoundingBox& bb) { return bb.max_x; });
                auto [miny1, miny2, miny_c] =
                    two_smallest([&](const BoundingBox& bb) { return bb.min_y; });
                auto [maxy1, maxy2, maxy_c] =
                    two_largest([&](const BoundingBox& bb) { return bb.max_y; });

                const double width0 = gmxx - gmnx;
                const double height0 = gmxy - gmny;
                const bool dom_x = (width0 >= height0);

                auto is_left = [&](int idx) -> bool {
                    return bbs[static_cast<size_t>(idx)].min_x <= minx1 + tol;
                };
                auto is_right = [&](int idx) -> bool {
                    return bbs[static_cast<size_t>(idx)].max_x >= maxx1 - tol;
                };
                auto is_bottom = [&](int idx) -> bool {
                    return bbs[static_cast<size_t>(idx)].min_y <= miny1 + tol;
                };
                auto is_top = [&](int idx) -> bool {
                    return bbs[static_cast<size_t>(idx)].max_y >= maxy1 - tol;
                };

                std::vector<double> base_gain(static_cast<size_t>(n), 0.0);
                for (int idx : boundary) {
                    const auto& bb = bbs[static_cast<size_t>(idx)];
                    double new_min_x = minx1;
                    if (minx_c == 1 && std::abs(bb.min_x - minx1) <= tol) {
                        new_min_x = minx2;
                    }
                    double new_max_x = maxx1;
                    if (maxx_c == 1 && std::abs(bb.max_x - maxx1) <= tol) {
                        new_max_x = maxx2;
                    }
                    double new_min_y = miny1;
                    if (miny_c == 1 && std::abs(bb.min_y - miny1) <= tol) {
                        new_min_y = miny2;
                    }
                    double new_max_y = maxy1;
                    if (maxy_c == 1 && std::abs(bb.max_y - maxy1) <= tol) {
                        new_max_y = maxy2;
                    }
                    double new_width = new_max_x - new_min_x;
                    double new_height = new_max_y - new_min_y;
                    double new_side = std::max(new_width, new_height);
                    double gain = curr_side - new_side;
                    if (gain > 0.0) {
                        base_gain[static_cast<size_t>(idx)] = gain;
                    }
                }

                auto pick_lns_remove_set = [&](const std::vector<int>& boundary_in,
                                               int want_remove) -> std::vector<int> {
                    if (static_cast<int>(boundary_in.size()) <= want_remove) {
                        return boundary_in;
                    }
                    if (want_remove <= 0) {
                        return {};
                    }

                    struct ScoredIdx {
                        double score;
                        int idx;
                    };
                    std::vector<ScoredIdx> scored;
                    scored.reserve(boundary_in.size());
                    double best_gain = 0.0;
                    for (int idx : boundary_in) {
                        double gain = base_gain[static_cast<size_t>(idx)];
                        if (gain > best_gain) {
                            best_gain = gain;
                        }
                        double score = (gain > 0.0) ? (gain + uni(rng) * 1e-12) : 0.0;
                        scored.push_back(ScoredIdx{score, idx});
                    }

                    if (!(best_gain > 0.0)) {
                        std::vector<int> out = boundary_in;
                        std::shuffle(out.begin(), out.end(), rng);
                        out.resize(static_cast<size_t>(want_remove));
                        return out;
                    }

                    std::sort(scored.begin(), scored.end(), [](const ScoredIdx& a, const ScoredIdx& b) {
                        if (a.score != b.score) {
                            return a.score > b.score;
                        }
                        return a.idx < b.idx;
                    });

                    std::vector<char> used(static_cast<size_t>(n), 0);
                    std::vector<int> out;
                    out.reserve(static_cast<size_t>(want_remove));

                    auto take_best = [&](auto pred) {
                        if (static_cast<int>(out.size()) >= want_remove) {
                            return;
                        }
                        for (const auto& si : scored) {
                            int idx = si.idx;
                            if (used[static_cast<size_t>(idx)]) {
                                continue;
                            }
                            if (!pred(idx)) {
                                continue;
                            }
                            used[static_cast<size_t>(idx)] = 1;
                            out.push_back(idx);
                            break;
                        }
                    };

                    // Garante que o LNS mexa em ambos os lados do eixo dominante (casca).
                    if (dom_x) {
                        take_best(is_left);
                        take_best(is_right);
                    } else {
                        take_best(is_bottom);
                        take_best(is_top);
                    }

                    for (const auto& si : scored) {
                        if (static_cast<int>(out.size()) >= want_remove) {
                            break;
                        }
                        int idx = si.idx;
                        if (used[static_cast<size_t>(idx)]) {
                            continue;
                        }
                        used[static_cast<size_t>(idx)] = 1;
                        out.push_back(idx);
                    }

                    if (static_cast<int>(out.size()) > want_remove) {
                        out.resize(static_cast<size_t>(want_remove));
                    }
                    return out;
                };

                std::vector<int> left;
                std::vector<int> right;
                std::vector<int> top;
                std::vector<int> bottom;
                std::vector<int> corners;
                left.reserve(boundary.size());
                right.reserve(boundary.size());
                top.reserve(boundary.size());
                bottom.reserve(boundary.size());
                corners.reserve(boundary.size());
                for (int idx : boundary) {
                    const bool l = is_left(idx);
                    const bool r = is_right(idx);
                    const bool b = is_bottom(idx);
                    const bool t = is_top(idx);
                    if (l) {
                        left.push_back(idx);
                    }
                    if (r) {
                        right.push_back(idx);
                    }
                    if (b) {
                        bottom.push_back(idx);
                    }
                    if (t) {
                        top.push_back(idx);
                    }
                    if ((l || r) && (b || t)) {
                        corners.push_back(idx);
                    }
                }

                auto fill_random_from = [&](std::vector<int>& out,
                                            const std::vector<int>& pool,
                                            int want) {
                    if (static_cast<int>(out.size()) >= want || pool.empty()) {
                        return;
                    }
                    std::vector<int> shuffled = pool;
                    std::shuffle(shuffled.begin(), shuffled.end(), rng);
                    std::vector<char> used(static_cast<size_t>(n), 0);
                    for (int idx : out) {
                        used[static_cast<size_t>(idx)] = 1;
                    }
                    for (int idx : shuffled) {
                        if (used[static_cast<size_t>(idx)]) {
                            continue;
                        }
                        out.push_back(idx);
                        if (static_cast<int>(out.size()) >= want) {
                            break;
                        }
                    }
                };

                auto pick_cluster_from_seed = [&](int seed,
                                                  const std::vector<int>& pool,
                                                  int want) -> std::vector<int> {
                    if (pool.empty() || want <= 0) {
                        return {};
                    }
                    std::vector<char> in_pool(static_cast<size_t>(n), 0);
                    for (int idx : pool) {
                        in_pool[static_cast<size_t>(idx)] = 1;
                    }
                    if (!in_pool[static_cast<size_t>(seed)]) {
                        seed = pool.front();
                    }

                    std::vector<char> used(static_cast<size_t>(n), 0);
                    std::vector<int> out;
                    out.reserve(static_cast<size_t>(want));
                    std::vector<int> queue;
                    queue.reserve(pool.size());
                    queue.push_back(seed);
                    used[static_cast<size_t>(seed)] = 1;
                    out.push_back(seed);

                    size_t qpos = 0;
                    while (qpos < queue.size() && static_cast<int>(out.size()) < want) {
                        int cur = queue[qpos++];
                        grid.gather(poses[static_cast<size_t>(cur)].x,
                                    poses[static_cast<size_t>(cur)].y,
                                    neigh);
                        for (int j : neigh) {
                            if (!in_pool[static_cast<size_t>(j)]) {
                                continue;
                            }
                            if (used[static_cast<size_t>(j)]) {
                                continue;
                            }
                            double dx = poses[static_cast<size_t>(j)].x -
                                        poses[static_cast<size_t>(cur)].x;
                            double dy = poses[static_cast<size_t>(j)].y -
                                        poses[static_cast<size_t>(cur)].y;
                            if (dx * dx + dy * dy > thr_sq) {
                                continue;
                            }
                            used[static_cast<size_t>(j)] = 1;
                            out.push_back(j);
                            queue.push_back(j);
                            if (static_cast<int>(out.size()) >= want) {
                                break;
                            }
                        }
                    }

                    if (static_cast<int>(out.size()) < want) {
                        std::vector<int> fallback = pool;
                        std::shuffle(fallback.begin(), fallback.end(), rng);
                        for (int idx : fallback) {
                            if (used[static_cast<size_t>(idx)]) {
                                continue;
                            }
                            out.push_back(idx);
                            if (static_cast<int>(out.size()) >= want) {
                                break;
                            }
                        }
                    }
                    return out;
                };

                auto pick_random_boundary = [&]() -> std::vector<int> {
                    std::vector<int> out = boundary;
                    std::shuffle(out.begin(), out.end(), rng);
                    if (static_cast<int>(out.size()) > m) {
                        out.resize(static_cast<size_t>(m));
                    }
                    return out;
                };

                auto pick_corner_candidate = [&]() -> std::vector<int> {
                    if (corners.empty()) {
                        return {};
                    }
                    std::vector<int> out = corners;
                    std::sort(out.begin(), out.end(), [&](int a, int b) {
                        double ga = base_gain[static_cast<size_t>(a)];
                        double gb = base_gain[static_cast<size_t>(b)];
                        if (ga != gb) {
                            return ga > gb;
                        }
                        return a < b;
                    });
                    if (static_cast<int>(out.size()) > m) {
                        out.resize(static_cast<size_t>(m));
                    }
                    return out;
                };

                auto pick_dual_cluster = [&]() -> std::vector<int> {
                    if (boundary.empty()) {
                        return {};
                    }
                    if (m <= 1) {
                        return pick_lns_remove_set(boundary, m);
                    }
                    const int m_left = m / 2;
                    const int m_right = m - m_left;
                    const std::vector<int>& pool_left = dom_x ? left : bottom;
                    const std::vector<int>& pool_right = dom_x ? right : top;
                    const std::vector<int>& pool_a = pool_left.empty() ? boundary : pool_left;
                    const std::vector<int>& pool_b = pool_right.empty() ? boundary : pool_right;
                    std::uniform_int_distribution<int> pick_a(0, static_cast<int>(pool_a.size()) - 1);
                    std::uniform_int_distribution<int> pick_b(0, static_cast<int>(pool_b.size()) - 1);
                    int seed_a = pool_a[static_cast<size_t>(pick_a(rng))];
                    int seed_b = pool_b[static_cast<size_t>(pick_b(rng))];
                    std::vector<int> out = pick_cluster_from_seed(seed_a, pool_a, m_left);
                    std::vector<int> other = pick_cluster_from_seed(seed_b, pool_b, m_right);
                    out.insert(out.end(), other.begin(), other.end());
                    return out;
                };

                std::vector<std::vector<int>> candidates;
                candidates.reserve(static_cast<size_t>(p.lns_candidates));
                auto add_candidate = [&](std::vector<int> cand) {
                    if (cand.empty()) {
                        return;
                    }
                    std::sort(cand.begin(), cand.end());
                    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
                    if (static_cast<int>(cand.size()) > m) {
                        cand.resize(static_cast<size_t>(m));
                    }
                    if (static_cast<int>(cand.size()) < m) {
                        fill_random_from(cand, boundary, m);
                    }
                    std::sort(cand.begin(), cand.end());
                    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
                    if (static_cast<int>(cand.size()) < m) {
                        fill_random_from(cand, boundary, m);
                    }
                    for (const auto& existing : candidates) {
                        if (existing == cand) {
                            return;
                        }
                    }
                    candidates.push_back(std::move(cand));
                };

                add_candidate(pick_lns_remove_set(boundary, m));
                if (static_cast<int>(candidates.size()) < p.lns_candidates) {
                    std::uniform_int_distribution<int> pick_seed(
                        0, static_cast<int>(boundary.size()) - 1);
                    int seed = boundary[static_cast<size_t>(pick_seed(rng))];
                    add_candidate(pick_cluster_from_seed(seed, boundary, m));
                }
                if (static_cast<int>(candidates.size()) < p.lns_candidates) {
                    add_candidate(pick_dual_cluster());
                }
                if (static_cast<int>(candidates.size()) < p.lns_candidates) {
                    add_candidate(pick_corner_candidate());
                }
                if (static_cast<int>(candidates.size()) < p.lns_candidates) {
                    add_candidate(pick_random_boundary());
                }

                int cand_attempts = 0;
                const int cand_attempts_max = std::max(4, p.lns_candidates * 6);
                while (static_cast<int>(candidates.size()) < p.lns_candidates &&
                       cand_attempts < cand_attempts_max) {
                    std::uniform_int_distribution<int> pick_seed(
                        0, static_cast<int>(boundary.size()) - 1);
                    int seed = boundary[static_cast<size_t>(pick_seed(rng))];
                    add_candidate(pick_cluster_from_seed(seed, boundary, m));
                    cand_attempts += 1;
                }
                if (candidates.empty()) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                struct LnsCandidateResult {
                    bool ok = false;
                    double min_x = 0.0;
                    double max_x = 0.0;
                    double min_y = 0.0;
                    double max_y = 0.0;
                    double new_width = 0.0;
                    double new_height = 0.0;
                    double new_side = std::numeric_limits<double>::infinity();
                    double new_min_dim = std::numeric_limits<double>::infinity();
                    double new_overlap = 0.0;
                    double new_cost = std::numeric_limits<double>::infinity();
                    std::vector<TreePose> poses;
                    std::vector<Polygon> polys;
                    std::vector<BoundingBox> bbs;
                };

                auto run_lns_candidate = [&](const std::vector<int>& remove_set,
                                             int attempts_per_tree) -> LnsCandidateResult {
                    LnsCandidateResult out;
                    if (remove_set.empty()) {
                        return out;
                    }

                    std::vector<char> active(static_cast<size_t>(n), 1);
                    for (int idx : remove_set) {
                        active[static_cast<size_t>(idx)] = 0;
                    }

                    double min_x = std::numeric_limits<double>::infinity();
                    double max_x = -std::numeric_limits<double>::infinity();
                    double min_y = std::numeric_limits<double>::infinity();
                    double max_y = -std::numeric_limits<double>::infinity();
                    for (int i = 0; i < n; ++i) {
                        if (!active[static_cast<size_t>(i)]) {
                            continue;
                        }
                        const auto& bb = bbs[static_cast<size_t>(i)];
                        min_x = std::min(min_x, bb.min_x);
                        max_x = std::max(max_x, bb.max_x);
                        min_y = std::min(min_y, bb.min_y);
                        max_y = std::max(max_y, bb.max_y);
                    }
                    if (!std::isfinite(min_x)) {
                        return out;
                    }

                    double box_side =
                        std::max(max_x - min_x, max_y - min_y) * std::max(1e-6, p.lns_box_mult);
                    const double ccx = 0.5 * (min_x + max_x);
                    const double ccy = 0.5 * (min_y + max_y);
                    const double half = 0.5 * box_side;
                    const double noise =
                        p.lns_noise_frac * std::max(1e-9, curr_side) * step_mult;

                    std::vector<TreePose> cand_poses = poses;
                    std::vector<Polygon> cand_polys = polys;
                    std::vector<BoundingBox> cand_bbs = bbs;

                    sa_refine::UniformGrid lns_grid(n, thr);
                    for (int j = 0; j < n; ++j) {
                        if (!active[static_cast<size_t>(j)]) {
                            continue;
                        }
                        lns_grid.insert(j,
                                        cand_poses[static_cast<size_t>(j)].x,
                                        cand_poses[static_cast<size_t>(j)].y);
                    }

                    double active_overlap = 0.0;
                    if (soft_overlap) {
                        for (int i = 0; i < n; ++i) {
                            if (!active[static_cast<size_t>(i)]) {
                                continue;
                            }
                            lns_grid.gather(cand_poses[static_cast<size_t>(i)].x,
                                            cand_poses[static_cast<size_t>(i)].y,
                                            neigh);
                            std::sort(neigh.begin(), neigh.end());
                            for (int j : neigh) {
                                if (j <= i) {
                                    continue;
                                }
                                double dx = cand_poses[static_cast<size_t>(i)].x -
                                            cand_poses[static_cast<size_t>(j)].x;
                                double dy = cand_poses[static_cast<size_t>(i)].y -
                                            cand_poses[static_cast<size_t>(j)].y;
                                if (dx * dx + dy * dy > thr_sq) {
                                    continue;
                                }
                                if (!aabb_overlap(cand_bbs[static_cast<size_t>(i)],
                                                  cand_bbs[static_cast<size_t>(j)])) {
                                    continue;
                                }
                                if (!polygons_intersect(cand_polys[static_cast<size_t>(i)],
                                                       cand_polys[static_cast<size_t>(j)])) {
                                    continue;
                                }
                                active_overlap += clamp_overlap(
                                    overlap_metric(cand_poses[static_cast<size_t>(i)],
                                                   cand_poses[static_cast<size_t>(j)]));
                            }
                        }
                        active_overlap = clamp_overlap(active_overlap);
                    }

                    std::vector<int> remove = remove_set;
                    std::shuffle(remove.begin(), remove.end(), rng);
                    const int max_attempts = std::max(1, attempts_per_tree);
                    for (int idx : remove) {
                        TreePose best_pose;
                        Polygon best_poly;
                        BoundingBox best_bb;
                        double best_metric = std::numeric_limits<double>::infinity();
                        double best_overlap_add = 0.0;
                        bool found = false;

                        for (int attempt = 0; attempt < max_attempts; ++attempt) {
                            TreePose cand = poses[static_cast<size_t>(idx)];
                            const double mode = uni(rng);

                            const double boundary_tol = 1e-9;
                            const auto& bb0 = bbs[static_cast<size_t>(idx)];
                            double pull = p.lns_pull_min +
                                          (p.lns_pull_max - p.lns_pull_min) * uni(rng);
                            if (mode < p.lns_p_uniform) {
                                // Uniform, mas com leve viés pra dentro (evita jogar a árvore de volta na casca).
                                double ux = (2.0 * uni(rng) - 1.0);
                                double uy = (2.0 * uni(rng) - 1.0);
                                cand.x = ccx + ux * (0.85 * half) + normal(rng) * (0.25 * noise);
                                cand.y = ccy + uy * (0.85 * half) + normal(rng) * (0.25 * noise);
                            } else if (mode < p.lns_p_uniform + p.lns_p_contact) {
                                // Contact: tenta ancorar em uma árvore ativa, mas orienta o ângulo pra "dentro".
                                int other = -1;
                                std::uniform_int_distribution<int> pick_any(0, n - 1);
                                for (int tries = 0; tries < 12; ++tries) {
                                    int j = pick_any(rng);
                                    if (j == idx) {
                                        continue;
                                    }
                                    if (!active[static_cast<size_t>(j)]) {
                                        continue;
                                    }
                                    other = j;
                                    break;
                                }
                                if (other < 0) {
                                    cand.x += pull * (ccx - cand.x) + normal(rng) * noise;
                                    cand.y += pull * (ccy - cand.y) + normal(rng) * noise;
                                } else {
                                    const double dist = thr * (0.88 + 0.14 * uni(rng));
                                    double base_ang = std::atan2(
                                        ccy - cand_poses[static_cast<size_t>(other)].y,
                                        ccx - cand_poses[static_cast<size_t>(other)].x);
                                    double jitter =
                                        (uni(rng) * 2.0 - 1.0) *
                                        (35.0 * 3.14159265358979323846 / 180.0);
                                    double ang = base_ang + jitter;
                                    cand.x = cand_poses[static_cast<size_t>(other)].x +
                                             dist * std::cos(ang) + normal(rng) * (0.15 * noise);
                                    cand.y = cand_poses[static_cast<size_t>(other)].y +
                                             dist * std::sin(ang) + normal(rng) * (0.15 * noise);
                                }
                            } else {
                                // Pull: como relocate, mas se idx estava na casca, puxa mais no eixo dominante.
                                cand.x += pull * (ccx - cand.x) + normal(rng) * noise;
                                cand.y += pull * (ccy - cand.y) + normal(rng) * noise;
                                if (dom_x) {
                                    if (bb0.min_x <= gmnx + boundary_tol) {
                                        cand.x += std::abs(normal(rng)) * (0.25 * half);
                                    } else if (bb0.max_x >= gmxx - boundary_tol) {
                                        cand.x -= std::abs(normal(rng)) * (0.25 * half);
                                    }
                                } else {
                                    if (bb0.min_y <= gmny + boundary_tol) {
                                        cand.y += std::abs(normal(rng)) * (0.25 * half);
                                    } else if (bb0.max_y >= gmxy - boundary_tol) {
                                        cand.y -= std::abs(normal(rng)) * (0.25 * half);
                                    }
                                }
                            }
                            if (uni(rng) < p.lns_p_rot) {
                                cand.deg = wrap_deg(cand.deg + uni_deg(rng) * ddeg_rng);
                            }
                            quantize_pose_inplace(cand);

                            if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 ||
                                cand.y > 100.0) {
                                continue;
                            }

                            Polygon poly = transform_polygon(base_poly_, cand);
                            BoundingBox bb = bounding_box(poly);

                            double overlap_add = 0.0;
                            bool collide = false;
                            lns_grid.gather(cand.x, cand.y, neigh);
                            for (int j : neigh) {
                                if (j == idx) {
                                    continue;
                                }
                                if (!active[static_cast<size_t>(j)]) {
                                    continue;
                                }
                                double ddx = cand.x - cand_poses[static_cast<size_t>(j)].x;
                                double ddy = cand.y - cand_poses[static_cast<size_t>(j)].y;
                                if (ddx * ddx + ddy * ddy > thr_sq) {
                                    continue;
                                }
                                if (!aabb_overlap(bb, cand_bbs[static_cast<size_t>(j)])) {
                                    continue;
                                }
                                if (polygons_intersect(poly, cand_polys[static_cast<size_t>(j)])) {
                                    if (!soft_overlap) {
                                        collide = true;
                                        break;
                                    }
                                    overlap_add += clamp_overlap(
                                        overlap_metric(cand, cand_poses[static_cast<size_t>(j)]));
                                }
                            }
                            if (collide) {
                                continue;
                            }

                            double nmin_x = std::min(min_x, bb.min_x);
                            double nmax_x = std::max(max_x, bb.max_x);
                            double nmin_y = std::min(min_y, bb.min_y);
                            double nmax_y = std::max(max_y, bb.max_y);
                            double cand_width = nmax_x - nmin_x;
                            double cand_height = nmax_y - nmin_y;
                            double total_overlap =
                                soft_overlap ? clamp_overlap(active_overlap + overlap_add) : 0.0;
                            double metric = cost_from(cand_width, cand_height, total_overlap);
                            if (soft_overlap && p.overlap_cost_cap > 0.0 &&
                                metric > p.overlap_cost_cap) {
                                continue;
                            }
                            if (metric + 1e-15 < best_metric) {
                                best_metric = metric;
                                best_overlap_add = overlap_add;
                                best_pose = cand;
                                best_poly = std::move(poly);
                                best_bb = bb;
                                found = true;
                            }
                        }

                        if (!found) {
                            return out;
                        }

                        cand_poses[static_cast<size_t>(idx)] = best_pose;
                        cand_polys[static_cast<size_t>(idx)] = std::move(best_poly);
                        cand_bbs[static_cast<size_t>(idx)] = best_bb;
                        active[static_cast<size_t>(idx)] = 1;
                        lns_grid.insert(idx, best_pose.x, best_pose.y);
                        if (soft_overlap) {
                            active_overlap = clamp_overlap(active_overlap + best_overlap_add);
                        }
                        min_x = std::min(min_x, best_bb.min_x);
                        max_x = std::max(max_x, best_bb.max_x);
                        min_y = std::min(min_y, best_bb.min_y);
                        max_y = std::max(max_y, best_bb.max_y);
                    }

                    const double new_width = max_x - min_x;
                    const double new_height = max_y - min_y;
                    double new_overlap = soft_overlap ? active_overlap : 0.0;
                    new_overlap = clamp_overlap(new_overlap);
                    double new_cost = cost_from(new_width, new_height, new_overlap);
                    if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                        return out;
                    }

                    out.ok = true;
                    out.min_x = min_x;
                    out.max_x = max_x;
                    out.min_y = min_y;
                    out.max_y = max_y;
                    out.new_width = new_width;
                    out.new_height = new_height;
                    out.new_min_dim = std::min(new_width, new_height);
                    out.new_side = std::max(new_width, new_height);
                    out.new_overlap = new_overlap;
                    out.new_cost = new_cost;
                    out.poses = std::move(cand_poses);
                    out.polys = std::move(cand_polys);
                    out.bbs = std::move(cand_bbs);
                    return out;
                };

                const int eval_attempts = (p.lns_eval_attempts_per_tree > 0)
                                              ? p.lns_eval_attempts_per_tree
                                              : p.lns_attempts_per_tree;
                bool found_candidate = false;
                std::vector<int> best_remove;
                LnsCandidateResult best_result;
                for (const auto& cand : candidates) {
                    LnsCandidateResult res = run_lns_candidate(cand, eval_attempts);
                    if (!res.ok) {
                        continue;
                    }
                    if (!found_candidate || res.new_cost + 1e-15 < best_result.new_cost) {
                        best_result = std::move(res);
                        best_remove = cand;
                        found_candidate = true;
                    }
                }

                if (!found_candidate) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                LnsCandidateResult final_result = std::move(best_result);
                if (eval_attempts != p.lns_attempts_per_tree) {
                    final_result = run_lns_candidate(best_remove, p.lns_attempts_per_tree);
                    if (!final_result.ok) {
                        add_reward(op, false, false, false, curr_cost, curr_cost);
                        maybe_update_controller(t);
                        continue;
                    }
                }

                const double new_width = final_result.new_width;
                const double new_height = final_result.new_height;
                const double new_min_dim = final_result.new_min_dim;
                const double new_side = final_result.new_side;
                const double new_overlap = final_result.new_overlap;
                const double new_cost = final_result.new_cost;
                const double min_x = final_result.min_x;
                const double max_x = final_result.max_x;
                const double min_y = final_result.min_y;
                const double max_y = final_result.max_y;

                reward_old_cost = curr_cost;
                reward_new_cost = new_cost;
                accepted = accept_move(curr_cost, new_cost, T);
                if (accepted) {
                    improved_curr = (new_cost + 1e-15 < curr_cost);
                    const bool valid = (new_overlap <= p.overlap_eps_area);
                    improved_best = valid && better_than_best(new_side, new_min_dim);
                    poses = std::move(final_result.poses);
                    polys = std::move(final_result.polys);
                    bbs = std::move(final_result.bbs);
                    grid.rebuild(poses);
                    gmnx = min_x;
                    gmxx = max_x;
                    gmny = min_y;
                    gmxy = max_y;
                    curr_side = new_side;
                    curr_overlap = new_overlap;
                    curr_cost = new_cost;
                    if (improved_best) {
                        best.best_side = curr_side;
                        best_min_dim = new_min_dim;
                        best.best_poses = poses;
                    }
                }
            
