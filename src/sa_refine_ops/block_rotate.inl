
                if (n <= 1) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }
                int anchor = picker.pick_index(1.0).idx;
                std::vector<int> block = build_block(anchor, p.block_size);
                std::vector<char> in_block(static_cast<size_t>(n), 0);
                for (int idx : block) {
                    in_block[static_cast<size_t>(idx)] = 1;
                }

                double px = 0.0;
                double py = 0.0;
                for (int idx : block) {
                    px += poses[static_cast<size_t>(idx)].x;
                    py += poses[static_cast<size_t>(idx)].y;
                }
                px /= static_cast<double>(block.size());
                py /= static_cast<double>(block.size());

                double rot_rng =
                    p.block_rot_deg_max * (1.0 - frac) + p.block_rot_deg_min * frac;
                rot_rng = std::min(180.0, rot_rng * step_mult);
                const double ang = uni_deg(rng) * rot_rng;
                const double rad = ang * 3.14159265358979323846 / 180.0;
                const double cA = std::cos(rad);
                const double sA = std::sin(rad);

                std::vector<TreePose> moved_pose;
                std::vector<Polygon> moved_poly;
                std::vector<BoundingBox> moved_bb;
                moved_pose.reserve(block.size());
                moved_poly.reserve(block.size());
                moved_bb.reserve(block.size());

                bool ok = true;
                for (int idx : block) {
                    const auto& src = poses[static_cast<size_t>(idx)];
                    double dx0 = src.x - px;
                    double dy0 = src.y - py;
	                    TreePose cand = src;
	                    cand.x = px + cA * dx0 - sA * dy0;
	                    cand.y = py + sA * dx0 + cA * dy0;
	                    cand.deg = wrap_deg(cand.deg + ang);
	                    quantize_pose_inplace(cand);
	                    if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
	                        ok = false;
	                        break;
	                    }
                    Polygon poly = transform_polygon(base_poly_, cand);
                    BoundingBox bb = bounding_box(poly);
                    moved_pose.push_back(cand);
                    moved_poly.push_back(std::move(poly));
                    moved_bb.push_back(bb);
                }
                if (!ok) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                if (!soft_overlap) {
                    for (size_t bi = 0; bi < block.size() && ok; ++bi) {
                        const auto& cand = moved_pose[bi];
                        const auto& cand_bb = moved_bb[bi];
                        const auto& cand_poly = moved_poly[bi];
                        grid.gather(cand.x, cand.y, neigh);
                        for (int j : neigh) {
                            if (in_block[static_cast<size_t>(j)]) {
                                continue;
                            }
                            double ddx = cand.x - poses[static_cast<size_t>(j)].x;
                            double ddy = cand.y - poses[static_cast<size_t>(j)].y;
                            if (ddx * ddx + ddy * ddy > thr_sq) {
                                continue;
                            }
                            if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                                continue;
                            }
                            if (polygons_intersect(cand_poly, polys[static_cast<size_t>(j)])) {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok) {
                            break;
                        }
                    }
                    if (!ok) {
                        add_reward(op, false, false, false, curr_cost, curr_cost);
                        maybe_update_controller(t);
                        continue;
                    }
                }

                Extents e2 = compute_extents_mixed(bbs, block, moved_bb);
                double new_side = side_from_extents(e2);

                const double old_side = curr_side;
                const double old_overlap = curr_overlap;
                const double old_cost = curr_cost;

                double delta_overlap = 0.0;
                if (soft_overlap) {
                    std::vector<int> pos(static_cast<size_t>(n), -1);
                    for (size_t bi = 0; bi < block.size(); ++bi) {
                        pos[static_cast<size_t>(block[bi])] = static_cast<int>(bi);
                    }

                    auto overlap_old = [&](int a, int b) -> double {
                        const double dx =
                            poses[static_cast<size_t>(a)].x - poses[static_cast<size_t>(b)].x;
                        const double dy =
                            poses[static_cast<size_t>(a)].y - poses[static_cast<size_t>(b)].y;
                        if (dx * dx + dy * dy > thr_sq) {
                            return 0.0;
                        }
                        if (!aabb_overlap(bbs[static_cast<size_t>(a)],
                                          bbs[static_cast<size_t>(b)]) ||
                            !polygons_intersect(polys[static_cast<size_t>(a)],
                                               polys[static_cast<size_t>(b)])) {
                            return 0.0;
                        }
                        return clamp_overlap(
                            overlap_metric(poses[static_cast<size_t>(a)],
                                           poses[static_cast<size_t>(b)]));
                    };
                    auto overlap_new = [&](int a, int b) -> double {
                        const int pa = pos[static_cast<size_t>(a)];
                        const int pb = pos[static_cast<size_t>(b)];
                        const TreePose& ta =
                            (pa >= 0) ? moved_pose[static_cast<size_t>(pa)]
                                      : poses[static_cast<size_t>(a)];
                        const TreePose& tb =
                            (pb >= 0) ? moved_pose[static_cast<size_t>(pb)]
                                      : poses[static_cast<size_t>(b)];
                        const double dx = ta.x - tb.x;
                        const double dy = ta.y - tb.y;
                        if (dx * dx + dy * dy > thr_sq) {
                            return 0.0;
                        }
                        const BoundingBox& ba =
                            (pa >= 0) ? moved_bb[static_cast<size_t>(pa)]
                                      : bbs[static_cast<size_t>(a)];
                        const BoundingBox& bb =
                            (pb >= 0) ? moved_bb[static_cast<size_t>(pb)]
                                      : bbs[static_cast<size_t>(b)];
                        const Polygon& pa_poly =
                            (pa >= 0) ? moved_poly[static_cast<size_t>(pa)]
                                      : polys[static_cast<size_t>(a)];
                        const Polygon& pb_poly =
                            (pb >= 0) ? moved_poly[static_cast<size_t>(pb)]
                                      : polys[static_cast<size_t>(b)];
                        if (!aabb_overlap(ba, bb) || !polygons_intersect(pa_poly, pb_poly)) {
                            return 0.0;
                        }
                        return clamp_overlap(overlap_metric(ta, tb));
                    };

                    for (int a : block) {
                        for (int b = 0; b < n; ++b) {
                            if (b == a) {
                                continue;
                            }
                            if (pos[static_cast<size_t>(b)] >= 0 && b < a) {
                                continue;
                            }
                            delta_overlap += overlap_new(a, b) - overlap_old(a, b);
                        }
                    }
                }

                double new_overlap = std::max(0.0, old_overlap + delta_overlap);
                new_overlap = clamp_overlap(new_overlap);
                const double new_width = e2.max_x - e2.min_x;
                const double new_height = e2.max_y - e2.min_y;
                const double new_min_dim = std::min(new_width, new_height);
                double new_cost = cost_from(new_width, new_height, new_overlap);
                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                    new_cost = std::numeric_limits<double>::infinity();
                }

                reward_old_cost = old_cost;
                reward_new_cost = new_cost;
                accepted = accept_move(old_cost, new_cost, T);
	                if (accepted) {
	                    improved_curr = (new_cost + 1e-15 < old_cost);
	                    const bool valid = (new_overlap <= p.overlap_eps_area);
	                    improved_best = valid && better_than_best(new_side, new_min_dim);
	                    for (size_t bi = 0; bi < block.size(); ++bi) {
	                        int idx = block[bi];
                            grid.update_position(idx,
                                                 moved_pose[bi].x,
                                                 moved_pose[bi].y);
	                        poses[static_cast<size_t>(idx)] = moved_pose[bi];
	                        polys[static_cast<size_t>(idx)] = std::move(moved_poly[bi]);
	                        bbs[static_cast<size_t>(idx)] = moved_bb[bi];
	                    }
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
	                } else {
	                    curr_side = old_side;
	                    curr_overlap = old_overlap;
                    curr_cost = old_cost;
                }
            