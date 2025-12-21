
                int i = picker.pick_index(p.p_pick_extreme).idx;
                int j = picker.pick_other_index(i);

	                TreePose cand_i = poses[static_cast<size_t>(i)];
	                TreePose cand_j = poses[static_cast<size_t>(j)];
	                std::swap(cand_i.deg, cand_j.deg);
	                quantize_pose_inplace(cand_i);
	                quantize_pose_inplace(cand_j);

		                Polygon poly_i = transform_polygon(base_poly_, cand_i);
		                Polygon poly_j = transform_polygon(base_poly_, cand_j);
		                BoundingBox bb_i = bounding_box(poly_i);
		                BoundingBox bb_j = bounding_box(poly_j);

	                bool ok = true;
	                if (!soft_overlap) {
	                    {
	                        double ddx = cand_i.x - cand_j.x;
	                        double ddy = cand_i.y - cand_j.y;
	                        if (ddx * ddx + ddy * ddy <= thr_sq) {
	                            if (aabb_overlap(bb_i, bb_j) &&
	                                polygons_intersect(poly_i, poly_j)) {
	                                ok = false;
	                            }
	                        }
	                    }

	                    if (ok) {
                            grid.gather(cand_i.x, cand_i.y, neigh);
                            for (int k : neigh) {
                                if (k == i || k == j) {
                                    continue;
                                }
                                double ddx = cand_i.x - poses[static_cast<size_t>(k)].x;
                                double ddy = cand_i.y - poses[static_cast<size_t>(k)].y;
                                if (ddx * ddx + ddy * ddy <= thr_sq) {
                                    if (aabb_overlap(bb_i, bbs[static_cast<size_t>(k)]) &&
                                        polygons_intersect(poly_i,
                                                           polys[static_cast<size_t>(k)])) {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                        }
                        if (ok) {
                            grid.gather(cand_j.x, cand_j.y, neigh2);
                            for (int k : neigh2) {
                                if (k == i || k == j) {
                                    continue;
                                }
                                double ddx = cand_j.x - poses[static_cast<size_t>(k)].x;
                                double ddy = cand_j.y - poses[static_cast<size_t>(k)].y;
                                if (ddx * ddx + ddy * ddy <= thr_sq) {
                                    if (aabb_overlap(bb_j, bbs[static_cast<size_t>(k)]) &&
                                        polygons_intersect(poly_j,
                                                           polys[static_cast<size_t>(k)])) {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
	                    }
	                }

	                if (!ok) {
	                    add_reward(op, false, false, false, curr_cost, curr_cost);
	                    maybe_update_controller(t);
	                    continue;
	                }

	                const BoundingBox old_bb_i = bbs[static_cast<size_t>(i)];
	                const BoundingBox old_bb_j = bbs[static_cast<size_t>(j)];
	                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
	                             old_gmxy = gmxy;
	                const double old_side = curr_side;
	                const double old_cost = curr_cost;
	                const double old_overlap = curr_overlap;

                bbs[static_cast<size_t>(i)] = bb_i;
                bbs[static_cast<size_t>(j)] = bb_j;
                gmnx = std::min(gmnx, std::min(bb_i.min_x, bb_j.min_x));
                gmxx = std::max(gmxx, std::max(bb_i.max_x, bb_j.max_x));
                gmny = std::min(gmny, std::min(bb_i.min_y, bb_j.min_y));
                gmxy = std::max(gmxy, std::max(bb_i.max_y, bb_j.max_y));

                bool need_full =
                    (old_bb_i.min_x <= old_gmnx + tol) ||
                    (old_bb_i.max_x >= old_gmxx - tol) ||
                    (old_bb_i.min_y <= old_gmny + tol) ||
                    (old_bb_i.max_y >= old_gmxy - tol) ||
                    (old_bb_j.min_x <= old_gmnx + tol) ||
                    (old_bb_j.max_x >= old_gmxx - tol) ||
                    (old_bb_j.min_y <= old_gmny + tol) ||
                    (old_bb_j.max_y >= old_gmxy - tol);
                if (need_full) {
                    Extents e2 = compute_extents(bbs);
                    gmnx = e2.min_x;
                    gmxx = e2.max_x;
                    gmny = e2.min_y;
                    gmxy = e2.max_y;
                }

                    const double new_width = gmxx - gmnx;
                    const double new_height = gmxy - gmny;
	                double new_side = std::max(new_width, new_height);
                    const double new_min_dim = std::min(new_width, new_height);
	                double delta_overlap = 0.0;
	                if (soft_overlap) {
	                    auto old_pair = [&](int a, int b) -> double {
	                        const double dx = poses[static_cast<size_t>(a)].x - poses[static_cast<size_t>(b)].x;
	                        const double dy = poses[static_cast<size_t>(a)].y - poses[static_cast<size_t>(b)].y;
	                        if (dx * dx + dy * dy > thr_sq) {
	                            return 0.0;
	                        }
	                        const BoundingBox& bb_a = (a == i) ? old_bb_i : (a == j ? old_bb_j : bbs[static_cast<size_t>(a)]);
	                        const BoundingBox& bb_b = (b == i) ? old_bb_i : (b == j ? old_bb_j : bbs[static_cast<size_t>(b)]);
	                        const Polygon& poly_a = polys[static_cast<size_t>(a)];
	                        const Polygon& poly_b = polys[static_cast<size_t>(b)];
	                        if (!aabb_overlap(bb_a, bb_b) || !polygons_intersect(poly_a, poly_b)) {
	                            return 0.0;
	                        }
	                        return clamp_overlap(
	                            overlap_metric(poses[static_cast<size_t>(a)],
	                                           poses[static_cast<size_t>(b)]));
	                    };
	                    auto new_pair = [&](int a, int b) -> double {
	                        const TreePose& pa = (a == i) ? cand_i : (a == j ? cand_j : poses[static_cast<size_t>(a)]);
	                        const TreePose& pb = (b == i) ? cand_i : (b == j ? cand_j : poses[static_cast<size_t>(b)]);
	                        const double dx = pa.x - pb.x;
	                        const double dy = pa.y - pb.y;
	                        if (dx * dx + dy * dy > thr_sq) {
	                            return 0.0;
	                        }
	                        const BoundingBox& bb_a = (a == i) ? bb_i : (a == j ? bb_j : bbs[static_cast<size_t>(a)]);
	                        const BoundingBox& bb_b = (b == i) ? bb_i : (b == j ? bb_j : bbs[static_cast<size_t>(b)]);
	                        const Polygon& poly_a = (a == i) ? poly_i : (a == j ? poly_j : polys[static_cast<size_t>(a)]);
	                        const Polygon& poly_b = (b == i) ? poly_i : (b == j ? poly_j : polys[static_cast<size_t>(b)]);
	                        if (!aabb_overlap(bb_a, bb_b) || !polygons_intersect(poly_a, poly_b)) {
	                            return 0.0;
	                        }
	                        return clamp_overlap(overlap_metric(pa, pb));
	                    };

	                    delta_overlap += new_pair(i, j) - old_pair(i, j);
                        gather_union(i,
                                     j,
                                     poses[static_cast<size_t>(i)].x,
                                     poses[static_cast<size_t>(i)].y,
                                     poses[static_cast<size_t>(j)].x,
                                     poses[static_cast<size_t>(j)].y,
                                     neigh_union);
	                    for (int k : neigh_union) {
	                        delta_overlap += new_pair(i, k) - old_pair(i, k);
	                        delta_overlap += new_pair(j, k) - old_pair(j, k);
	                    }
	                }
	                double new_overlap = std::max(0.0, old_overlap + delta_overlap);
	                new_overlap = clamp_overlap(new_overlap);
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
	                    poses[static_cast<size_t>(i)].deg = cand_i.deg;
	                    poses[static_cast<size_t>(j)].deg = cand_j.deg;
	                    polys[static_cast<size_t>(i)] = std::move(poly_i);
	                    polys[static_cast<size_t>(j)] = std::move(poly_j);
	                    curr_side = new_side;
	                    curr_overlap = new_overlap;
	                    curr_cost = new_cost;
	                    if (improved_best) {
	                        best.best_side = curr_side;
                            best_min_dim = new_min_dim;
	                        best.best_poses = poses;
	                    }
	                } else {
	                    bbs[static_cast<size_t>(i)] = old_bb_i;
	                    bbs[static_cast<size_t>(j)] = old_bb_j;
	                    gmnx = old_gmnx;
	                    gmxx = old_gmxx;
	                    gmny = old_gmny;
	                    gmxy = old_gmxy;
	                    curr_side = old_side;
	                    curr_overlap = old_overlap;
	                    curr_cost = old_cost;
	                }
	            