
                if (n <= 1 || !(p.global_rot_deg > 0.0)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }
                if (use_mask && static_cast<int>(active_indices.size()) != n) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const double ddeg = uni_deg(rng) * p.global_rot_deg;
                if (std::abs(ddeg) < 1e-12) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const double kPi = 3.14159265358979323846;
                const double rad = ddeg * kPi / 180.0;
                const double c = std::cos(rad);
                const double s = std::sin(rad);

                std::vector<TreePose> cand_poses = poses;
                bool ok_bounds = true;
                for (int i = 0; i < n; ++i) {
                    const double dx = poses[static_cast<size_t>(i)].x - cx;
                    const double dy = poses[static_cast<size_t>(i)].y - cy;
                    const double rx = c * dx - s * dy;
                    const double ry = s * dx + c * dy;
                    cand_poses[static_cast<size_t>(i)].x = cx + rx;
                    cand_poses[static_cast<size_t>(i)].y = cy + ry;
                    cand_poses[static_cast<size_t>(i)].deg =
                        wrap_deg(poses[static_cast<size_t>(i)].deg + ddeg);
                    quantize_pose_inplace(cand_poses[static_cast<size_t>(i)]);
                    if (cand_poses[static_cast<size_t>(i)].x < -100.0 ||
                        cand_poses[static_cast<size_t>(i)].x > 100.0 ||
                        cand_poses[static_cast<size_t>(i)].y < -100.0 ||
                        cand_poses[static_cast<size_t>(i)].y > 100.0) {
                        ok_bounds = false;
                        break;
                    }
                }
                if (!ok_bounds) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                std::vector<Polygon> cand_polys;
                std::vector<BoundingBox> cand_bbs;
                cand_polys.reserve(static_cast<size_t>(n));
                cand_bbs.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    Polygon poly = transform_polygon(base_poly_, cand_poses[static_cast<size_t>(i)]);
                    cand_polys.push_back(std::move(poly));
                    cand_bbs.push_back(bounding_box(cand_polys.back()));
                }

                bool ok = true;
                double new_overlap = 0.0;
                for (int i = 0; i < n && ok; ++i) {
                    for (int j = i + 1; j < n; ++j) {
                        const double dx = cand_poses[static_cast<size_t>(i)].x -
                                          cand_poses[static_cast<size_t>(j)].x;
                        const double dy = cand_poses[static_cast<size_t>(i)].y -
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
                        if (!soft_overlap) {
                            ok = false;
                            break;
                        }
                        new_overlap += clamp_overlap(
                            overlap_metric(cand_poses[static_cast<size_t>(i)],
                                           cand_poses[static_cast<size_t>(j)]));
                    }
                }
                if (!ok) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                new_overlap = clamp_overlap(new_overlap);
                Extents e2 = compute_extents(cand_bbs);
                const double new_width = e2.max_x - e2.min_x;
                const double new_height = e2.max_y - e2.min_y;
                const double new_min_dim = std::min(new_width, new_height);
                const double new_side = std::max(new_width, new_height);
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
                    gmnx = e2.min_x;
                    gmxx = e2.max_x;
                    gmny = e2.min_y;
                    gmxy = e2.max_y;
                    curr_side = new_side;
                    curr_overlap = new_overlap;
                    curr_cost = new_cost;
                    if (improved_best) {
                        best.best_side = curr_side;
                        best_min_dim = new_min_dim;
                        best.best_poses = poses;
                    }
                }
            