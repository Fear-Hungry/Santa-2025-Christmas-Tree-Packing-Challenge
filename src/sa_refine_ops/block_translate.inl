
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

                double block_step =
                    (p.block_step_frac_max * (1.0 - frac) + p.block_step_frac_min * frac) *
                    std::max(1e-9, curr_side);
                block_step *= step_mult;
                double dx = 0.0;
                double dy = 0.0;
                if (uni(rng) < p.block_p_random_dir) {
                    dx = normal(rng) * block_step;
                    dy = normal(rng) * block_step;
                } else {
                    const double width = gmxx - gmnx;
                    const double height = gmxy - gmny;
                    const auto& bb = bbs[static_cast<size_t>(anchor)];
                    const double boundary_tol = 1e-9;
                    double mag = std::abs(normal(rng)) * block_step;
                    if (width >= height) {
                        bool left = (bb.min_x <= gmnx + boundary_tol);
                        bool right = (bb.max_x >= gmxx - boundary_tol);
                        double dir = left ? 1.0 : (right ? -1.0 : (cx - poses[static_cast<size_t>(anchor)].x >= 0.0 ? 1.0 : -1.0));
                        dx = dir * mag;
                        dy = normal(rng) * (0.35 * block_step);
                    } else {
                        bool bottom = (bb.min_y <= gmny + boundary_tol);
                        bool top = (bb.max_y >= gmxy - boundary_tol);
                        double dir = bottom ? 1.0 : (top ? -1.0 : (cy - poses[static_cast<size_t>(anchor)].y >= 0.0 ? 1.0 : -1.0));
                        dx = normal(rng) * (0.35 * block_step);
                        dy = dir * mag;
                    }
                }

	                std::vector<TreePose> moved_pose;
	                std::vector<Polygon> moved_poly;
	                std::vector<BoundingBox> moved_bb;
	                moved_pose.reserve(block.size());
	                moved_poly.reserve(block.size());
	                moved_bb.reserve(block.size());

	                bool ok = false;
	                double scale = 1.0;
	                for (int bt = 0; bt < 6; ++bt) {
	                    moved_pose.clear();
	                    moved_poly.clear();
	                    moved_bb.clear();

	                    bool ok_local = true;
		                    for (int idx : block) {
		                        TreePose cand = poses[static_cast<size_t>(idx)];
		                        cand.x += dx * scale;
		                        cand.y += dy * scale;
		                        quantize_pose_inplace(cand);
		                        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 ||
		                            cand.y > 100.0) {
		                            ok_local = false;
		                            break;
	                        }
	                        Polygon poly = transform_polygon(base_poly_, cand);
	                        BoundingBox bb = bounding_box(poly);
	                        moved_pose.push_back(cand);
	                        moved_poly.push_back(std::move(poly));
	                        moved_bb.push_back(bb);
	                    }

		                    if (ok_local && !soft_overlap) {
		                        for (size_t bi = 0; bi < block.size() && ok_local; ++bi) {
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
                                        if (polygons_intersect(cand_poly,
                                                               polys[static_cast<size_t>(j)])) {
                                            ok_local = false;
                                            break;
                                        }
                                    }
	                        }
	                    }

	                    if (ok_local) {
	                        ok = true;
	                        break;
	                    }

	                    scale *= 0.5;
	                }

		                if (!ok) {
		                    add_reward(op, false, false, false, curr_cost, curr_cost);
		                    maybe_update_controller(t);
		                    continue;
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
	                        const double dx = poses[static_cast<size_t>(a)].x - poses[static_cast<size_t>(b)].x;
	                        const double dy = poses[static_cast<size_t>(a)].y - poses[static_cast<size_t>(b)].y;
	                        if (dx * dx + dy * dy > thr_sq) {
	                            return 0.0;
	                        }
	                        if (!aabb_overlap(bbs[static_cast<size_t>(a)], bbs[static_cast<size_t>(b)]) ||
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
	                            (pa >= 0) ? moved_pose[static_cast<size_t>(pa)] : poses[static_cast<size_t>(a)];
	                        const TreePose& tb =
	                            (pb >= 0) ? moved_pose[static_cast<size_t>(pb)] : poses[static_cast<size_t>(b)];
	                        const double dx = ta.x - tb.x;
	                        const double dy = ta.y - tb.y;
	                        if (dx * dx + dy * dy > thr_sq) {
	                            return 0.0;
	                        }
	                        const BoundingBox& ba =
	                            (pa >= 0) ? moved_bb[static_cast<size_t>(pa)] : bbs[static_cast<size_t>(a)];
	                        const BoundingBox& bb =
	                            (pb >= 0) ? moved_bb[static_cast<size_t>(pb)] : bbs[static_cast<size_t>(b)];
	                        const Polygon& pa_poly =
	                            (pa >= 0) ? moved_poly[static_cast<size_t>(pa)] : polys[static_cast<size_t>(a)];
	                        const Polygon& pb_poly =
	                            (pb >= 0) ? moved_poly[static_cast<size_t>(pb)] : polys[static_cast<size_t>(b)];
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
	                }
            