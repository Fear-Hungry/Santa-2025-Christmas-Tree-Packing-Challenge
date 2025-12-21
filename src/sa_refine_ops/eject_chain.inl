
                if (n <= 2 || p.eject_reinsert_attempts <= 0) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }
                if (use_mask && static_cast<int>(active_indices.size()) != n) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const double boundary_tol = 1e-9;
                std::vector<int> active_all;
                active_all.reserve(static_cast<size_t>(n));
                std::vector<int> interior;
                interior.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    if (!is_active(i)) {
                        continue;
                    }
                    active_all.push_back(i);
                    const auto& bb = bbs[static_cast<size_t>(i)];
                    if (bb.min_x > gmnx + boundary_tol && bb.max_x < gmxx - boundary_tol &&
                        bb.min_y > gmny + boundary_tol && bb.max_y < gmxy - boundary_tol) {
                        interior.push_back(i);
                    }
                }
                const std::vector<int>& pool = interior.empty() ? active_all : interior;
                if (pool.empty()) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                struct DistIdx {
                    double d2;
                    int idx;
                };
                std::vector<DistIdx> dist;
                dist.reserve(pool.size());
                for (int idx : pool) {
                    const double dx = poses[static_cast<size_t>(idx)].x - cx;
                    const double dy = poses[static_cast<size_t>(idx)].y - cy;
                    dist.push_back(DistIdx{dx * dx + dy * dy, idx});
                }
                int topk = std::min(static_cast<int>(dist.size()),
                                    std::max(1, p.eject_center_topk));
                std::nth_element(dist.begin(),
                                 dist.begin() + (topk - 1),
                                 dist.end(),
                                 [](const DistIdx& a, const DistIdx& b) {
                                     return a.d2 < b.d2;
                                 });
                std::uniform_int_distribution<int> pick_top(0, topk - 1);
                const int remove_idx = dist[static_cast<size_t>(pick_top(rng))].idx;

                std::vector<TreePose> cand_poses = poses;
                std::vector<Polygon> cand_polys = polys;
                std::vector<BoundingBox> cand_bbs = bbs;

                std::vector<char> active(static_cast<size_t>(n), 1);
                active[static_cast<size_t>(remove_idx)] = 0;
                std::vector<int> active_list;
                active_list.reserve(static_cast<size_t>(n - 1));
                for (int i = 0; i < n; ++i) {
                    if (active[static_cast<size_t>(i)]) {
                        active_list.push_back(i);
                    }
                }
                if (active_list.size() < 2) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                sa_refine::UniformGrid eject_grid(n, thr);
                for (int idx : active_list) {
                    eject_grid.insert(idx,
                                      cand_poses[static_cast<size_t>(idx)].x,
                                      cand_poses[static_cast<size_t>(idx)].y);
                }

                auto compute_active_extents = [&]() -> Extents {
                    Extents e2;
                    e2.min_x = std::numeric_limits<double>::infinity();
                    e2.max_x = -std::numeric_limits<double>::infinity();
                    e2.min_y = std::numeric_limits<double>::infinity();
                    e2.max_y = -std::numeric_limits<double>::infinity();
                    for (int idx : active_list) {
                        const auto& bb = cand_bbs[static_cast<size_t>(idx)];
                        e2.min_x = std::min(e2.min_x, bb.min_x);
                        e2.max_x = std::max(e2.max_x, bb.max_x);
                        e2.min_y = std::min(e2.min_y, bb.min_y);
                        e2.max_y = std::max(e2.max_y, bb.max_y);
                    }
                    return e2;
                };

                const int relax_iters = std::max(0, p.eject_relax_iters);
                for (int it = 0; it < relax_iters; ++it) {
                    Extents e2 = compute_active_extents();
                    if (!std::isfinite(e2.min_x)) {
                        break;
                    }
                    const double ccx = 0.5 * (e2.min_x + e2.max_x);
                    const double ccy = 0.5 * (e2.min_y + e2.max_y);
                    const double relax_side = side_from_extents(e2);
                    const double max_step =
                        p.eject_step_frac * std::max(1e-9, relax_side) * step_mult;

                    std::shuffle(active_list.begin(), active_list.end(), rng);
                    for (int idx : active_list) {
                        const double dx0 = ccx - cand_poses[static_cast<size_t>(idx)].x;
                        const double dy0 = ccy - cand_poses[static_cast<size_t>(idx)].y;
                        const double norm = std::hypot(dx0, dy0);
                        if (!(norm > 1e-12)) {
                            continue;
                        }
                        const double ux = dx0 / norm;
                        const double uy = dy0 / norm;

                        bool moved = false;
                        double scale = 1.0;
                        for (int bt = 0; bt < 6 && !moved; ++bt) {
                            TreePose cand = cand_poses[static_cast<size_t>(idx)];
                            cand.x += ux * max_step * scale;
                            cand.y += uy * max_step * scale;
                            quantize_pose_inplace(cand);

                            if (cand.x < -100.0 || cand.x > 100.0 ||
                                cand.y < -100.0 || cand.y > 100.0) {
                                scale *= 0.5;
                                continue;
                            }

                            Polygon poly = transform_polygon(base_poly_, cand);
                            BoundingBox bb = bounding_box(poly);
                            bool collide = false;
                            eject_grid.gather(cand.x, cand.y, neigh);
                            for (int j : neigh) {
                                if (j == idx || !active[static_cast<size_t>(j)]) {
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
                                    collide = true;
                                    break;
                                }
                            }
                            if (collide) {
                                scale *= 0.5;
                                continue;
                            }

                            cand_poses[static_cast<size_t>(idx)] = cand;
                            cand_polys[static_cast<size_t>(idx)] = std::move(poly);
                            cand_bbs[static_cast<size_t>(idx)] = bb;
                            eject_grid.update_position(idx, cand.x, cand.y);
                            moved = true;
                        }
                    }
                }

                Extents active_ext = compute_active_extents();
                if (!std::isfinite(active_ext.min_x)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                double active_overlap = 0.0;
                if (soft_overlap) {
                    for (size_t ai = 0; ai < active_list.size(); ++ai) {
                        int i = active_list[ai];
                        for (size_t aj = ai + 1; aj < active_list.size(); ++aj) {
                            int j = active_list[aj];
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

                int count_left = 0;
                int count_right = 0;
                int count_bottom = 0;
                int count_top = 0;
                for (int idx : active_list) {
                    const auto& bb = cand_bbs[static_cast<size_t>(idx)];
                    if (bb.min_x <= active_ext.min_x + boundary_tol) {
                        count_left += 1;
                    }
                    if (bb.max_x >= active_ext.max_x - boundary_tol) {
                        count_right += 1;
                    }
                    if (bb.min_y <= active_ext.min_y + boundary_tol) {
                        count_bottom += 1;
                    }
                    if (bb.max_y >= active_ext.max_y - boundary_tol) {
                        count_top += 1;
                    }
                }

                const double ccx = 0.5 * (active_ext.min_x + active_ext.max_x);
                const double ccy = 0.5 * (active_ext.min_y + active_ext.max_y);
                struct SidePick {
                    int count = 0;
                    double slack = 0.0;
                    int side = 0;
                };
                std::array<SidePick, 4> sides = {
                    SidePick{count_left, std::abs(ccx - active_ext.min_x), 0},
                    SidePick{count_right, std::abs(active_ext.max_x - ccx), 1},
                    SidePick{count_bottom, std::abs(ccy - active_ext.min_y), 2},
                    SidePick{count_top, std::abs(active_ext.max_y - ccy), 3},
                };
                std::sort(sides.begin(),
                          sides.end(),
                          [](const SidePick& a, const SidePick& b) {
                              if (a.count != b.count) {
                                  return a.count < b.count;
                              }
                              if (a.slack != b.slack) {
                                  return a.slack > b.slack;
                              }
                              return a.side < b.side;
                          });
                const int side = sides[0].side;

                const double noise =
                    p.eject_reinsert_noise_frac * std::max(1e-9, curr_side);
                TreePose best_pose;
                Polygon best_poly;
                BoundingBox best_bb;
                double best_overlap_add = 0.0;
                double best_cost = std::numeric_limits<double>::infinity();
                bool found = false;

                for (int att = 0; att < p.eject_reinsert_attempts; ++att) {
                    TreePose cand = poses[static_cast<size_t>(remove_idx)];
                    const double t = 0.25 + 0.75 * uni(rng);

                    if (side == 0) {
                        cand.x = active_ext.min_x - radius_ * t + normal(rng) * noise;
                        cand.y = active_ext.min_y +
                                 (active_ext.max_y - active_ext.min_y) * uni(rng) +
                                 normal(rng) * (0.25 * noise);
                    } else if (side == 1) {
                        cand.x = active_ext.max_x + radius_ * t + normal(rng) * noise;
                        cand.y = active_ext.min_y +
                                 (active_ext.max_y - active_ext.min_y) * uni(rng) +
                                 normal(rng) * (0.25 * noise);
                    } else if (side == 2) {
                        cand.y = active_ext.min_y - radius_ * t + normal(rng) * noise;
                        cand.x = active_ext.min_x +
                                 (active_ext.max_x - active_ext.min_x) * uni(rng) +
                                 normal(rng) * (0.25 * noise);
                    } else {
                        cand.y = active_ext.max_y + radius_ * t + normal(rng) * noise;
                        cand.x = active_ext.min_x +
                                 (active_ext.max_x - active_ext.min_x) * uni(rng) +
                                 normal(rng) * (0.25 * noise);
                    }

                    if (uni(rng) < p.eject_reinsert_p_rot &&
                        p.eject_reinsert_rot_deg > 0.0) {
                        cand.deg = wrap_deg(cand.deg + uni_deg(rng) * p.eject_reinsert_rot_deg);
                    }
                    quantize_pose_inplace(cand);
                    if (cand.x < -100.0 || cand.x > 100.0 ||
                        cand.y < -100.0 || cand.y > 100.0) {
                        continue;
                    }

                    Polygon poly = transform_polygon(base_poly_, cand);
                    BoundingBox bb = bounding_box(poly);
                    bool collide = false;
                    double overlap_add = 0.0;
                    for (int j : active_list) {
                        double dx = cand.x - cand_poses[static_cast<size_t>(j)].x;
                        double dy = cand.y - cand_poses[static_cast<size_t>(j)].y;
                        if (dx * dx + dy * dy > thr_sq) {
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

                    const double nmin_x = std::min(active_ext.min_x, bb.min_x);
                    const double nmax_x = std::max(active_ext.max_x, bb.max_x);
                    const double nmin_y = std::min(active_ext.min_y, bb.min_y);
                    const double nmax_y = std::max(active_ext.max_y, bb.max_y);
                    const double nwidth = nmax_x - nmin_x;
                    const double nheight = nmax_y - nmin_y;
                    const double total_overlap =
                        soft_overlap ? clamp_overlap(active_overlap + overlap_add) : 0.0;
                    double metric = cost_from(nwidth, nheight, total_overlap);
                    if (p.overlap_cost_cap > 0.0 && metric > p.overlap_cost_cap) {
                        continue;
                    }
                    if (metric + 1e-15 < best_cost) {
                        best_cost = metric;
                        best_overlap_add = overlap_add;
                        best_pose = cand;
                        best_poly = std::move(poly);
                        best_bb = bb;
                        found = true;
                    }
                }

                if (!found) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                cand_poses[static_cast<size_t>(remove_idx)] = best_pose;
                cand_polys[static_cast<size_t>(remove_idx)] = std::move(best_poly);
                cand_bbs[static_cast<size_t>(remove_idx)] = best_bb;

                const double new_min_x = std::min(active_ext.min_x, best_bb.min_x);
                const double new_max_x = std::max(active_ext.max_x, best_bb.max_x);
                const double new_min_y = std::min(active_ext.min_y, best_bb.min_y);
                const double new_max_y = std::max(active_ext.max_y, best_bb.max_y);
                const double new_width = new_max_x - new_min_x;
                const double new_height = new_max_y - new_min_y;
                const double new_min_dim = std::min(new_width, new_height);
                const double new_side = std::max(new_width, new_height);
                double new_overlap =
                    soft_overlap ? clamp_overlap(active_overlap + best_overlap_add) : 0.0;
                double new_cost = cost_from(new_width, new_height, new_overlap);
                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                reward_old_cost = curr_cost;
                reward_new_cost = new_cost;
                accepted = accept_move(curr_cost, new_cost, T);
                if (accepted) {
                    improved_curr = (new_cost + 1e-15 < curr_cost);
                    const bool valid = soft_overlap ? (new_overlap <= p.overlap_eps_area) : true;
                    improved_best = valid && better_than_best(new_side, new_min_dim);
                    poses = std::move(cand_poses);
                    polys = std::move(cand_polys);
                    bbs = std::move(cand_bbs);
                    grid.rebuild(poses);
                    gmnx = new_min_x;
                    gmxx = new_max_x;
                    gmny = new_min_y;
                    gmxy = new_max_y;
                    curr_side = new_side;
                    curr_overlap = new_overlap;
                    curr_cost = new_cost;
                    if (improved_best) {
                        best.best_side = curr_side;
                        best_min_dim = new_min_dim;
                        best.best_poses = poses;
                    }
                }
            
