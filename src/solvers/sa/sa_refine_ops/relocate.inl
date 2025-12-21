
	                int i = picker.pick_index(p.p_pick_extreme).idx;

	                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
	                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
	                             old_gmxy = gmxy;
	                const double old_side = curr_side;
	                const double old_overlap = curr_overlap;
	                const double old_cost = curr_cost;

	                const double noise =
                        p.relocate_noise_frac * std::max(1e-9, curr_side) * step_mult;

	                TreePose best_cand;
	                Polygon best_poly;
	                BoundingBox best_bb;
	                Extents best_extents;
	                double best_new_side = std::numeric_limits<double>::infinity();
                    double best_new_min_dim = std::numeric_limits<double>::infinity();
	                double best_new_overlap = old_overlap;
	                double best_new_cost = std::numeric_limits<double>::infinity();
	                bool found = false;

	                for (int attempt = 0; attempt < std::max(1, p.relocate_attempts); ++attempt) {
	                    double pull =
	                        p.relocate_pull_min +
	                        (p.relocate_pull_max - p.relocate_pull_min) * uni(rng);

	                    TreePose cand = poses[static_cast<size_t>(i)];
		                    cand.x += pull * (cx - cand.x) + normal(rng) * noise;
		                    cand.y += pull * (cy - cand.y) + normal(rng) * noise;
		                    if (uni(rng) < p.relocate_p_rot) {
		                        cand.deg = wrap_deg(cand.deg + uni_deg(rng) * ddeg_rng);
		                    }
		                    quantize_pose_inplace(cand);

		                    if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
		                        continue;
		                    }

	                    Polygon cand_poly = transform_polygon(base_poly_, cand);
	                    BoundingBox cand_bb = bounding_box(cand_poly);

		                    if (!soft_overlap) {
		                        bool ok = true;
                                grid.gather(cand.x, cand.y, neigh);
                                for (int k : neigh) {
                                    if (k == i) {
                                        continue;
                                    }
                                    double ddx = cand.x - poses[static_cast<size_t>(k)].x;
                                    double ddy = cand.y - poses[static_cast<size_t>(k)].y;
                                    if (ddx * ddx + ddy * ddy > thr_sq) {
                                        continue;
                                    }
                                    if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(k)])) {
                                        continue;
                                    }
                                    if (polygons_intersect(cand_poly, polys[static_cast<size_t>(k)])) {
                                        ok = false;
                                        break;
                                    }
                                }
		                        if (!ok) {
		                            continue;
		                        }

	                        // Hard mode: usa o caminho incremental original.
	                        bbs[static_cast<size_t>(i)] = cand_bb;
	                        gmnx = std::min(gmnx, cand_bb.min_x);
	                        gmxx = std::max(gmxx, cand_bb.max_x);
	                        gmny = std::min(gmny, cand_bb.min_y);
	                        gmxy = std::max(gmxy, cand_bb.max_y);

	                        bool need_full = (old_bb.min_x <= old_gmnx + tol) ||
	                                         (old_bb.max_x >= old_gmxx - tol) ||
	                                         (old_bb.min_y <= old_gmny + tol) ||
	                                         (old_bb.max_y >= old_gmxy - tol);
	                        if (need_full) {
	                            Extents e2 = compute_extents(bbs);
	                            gmnx = e2.min_x;
	                            gmxx = e2.max_x;
	                            gmny = e2.min_y;
	                            gmxy = e2.max_y;
	                        }

                            const double cand_width = gmxx - gmnx;
                            const double cand_height = gmxy - gmny;
	                        best_new_side = std::max(cand_width, cand_height);
                            best_new_min_dim = std::min(cand_width, cand_height);
	                        best_new_cost = cost_from(cand_width, cand_height, 0.0);
	                        best_new_overlap = 0.0;
	                        best_cand = cand;
	                        best_poly = std::move(cand_poly);
	                        best_bb = cand_bb;
	                        best_extents = Extents{gmnx, gmxx, gmny, gmxy};
	                        found = true;

	                        // restaura para avaliar acceptance fora.
	                        bbs[static_cast<size_t>(i)] = old_bb;
	                        gmnx = old_gmnx;
	                        gmxx = old_gmxx;
	                        gmny = old_gmny;
	                        gmxy = old_gmxy;
	                        break;
	                    }

		                    // Soft mode: escolhe o melhor candidato por custo.
		                    double delta_overlap = 0.0;
                            gather_union(i,
                                         -1,
                                         poses[static_cast<size_t>(i)].x,
                                         poses[static_cast<size_t>(i)].y,
                                         cand.x,
                                         cand.y,
                                         neigh_union);
		                    for (int k : neigh_union) {
		                        const double old_dx =
		                            poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(k)].x;
		                        const double old_dy =
		                            poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(k)].y;
	                        const double new_dx = cand.x - poses[static_cast<size_t>(k)].x;
	                        const double new_dy = cand.y - poses[static_cast<size_t>(k)].y;

	                        double old_a = 0.0;
	                        if ((old_dx * old_dx + old_dy * old_dy <= thr_sq) &&
	                            aabb_overlap(old_bb, bbs[static_cast<size_t>(k)]) &&
	                            polygons_intersect(polys[static_cast<size_t>(i)],
	                                               polys[static_cast<size_t>(k)])) {
	                            old_a = overlap_metric(poses[static_cast<size_t>(i)],
	                                                   poses[static_cast<size_t>(k)]);
	                        }
	                        double new_a = 0.0;
	                        if ((new_dx * new_dx + new_dy * new_dy <= thr_sq) &&
	                            aabb_overlap(cand_bb, bbs[static_cast<size_t>(k)]) &&
	                            polygons_intersect(cand_poly, polys[static_cast<size_t>(k)])) {
	                            new_a = overlap_metric(cand, poses[static_cast<size_t>(k)]);
	                        }
	                        delta_overlap += clamp_overlap(new_a) - clamp_overlap(old_a);
	                    }
	                    double new_overlap = std::max(0.0, old_overlap + delta_overlap);
	                    new_overlap = clamp_overlap(new_overlap);

	                    Extents e2;
	                    e2.min_x = std::numeric_limits<double>::infinity();
	                    e2.max_x = -std::numeric_limits<double>::infinity();
	                    e2.min_y = std::numeric_limits<double>::infinity();
	                    e2.max_y = -std::numeric_limits<double>::infinity();
	                    for (int k = 0; k < n; ++k) {
	                        const BoundingBox& bb = (k == i) ? cand_bb : bbs[static_cast<size_t>(k)];
	                        e2.min_x = std::min(e2.min_x, bb.min_x);
	                        e2.max_x = std::max(e2.max_x, bb.max_x);
	                        e2.min_y = std::min(e2.min_y, bb.min_y);
	                        e2.max_y = std::max(e2.max_y, bb.max_y);
	                    }
                        const double new_width = e2.max_x - e2.min_x;
                        const double new_height = e2.max_y - e2.min_y;
	                    double new_side = std::max(new_width, new_height);
                        const double new_min_dim = std::min(new_width, new_height);
	                    double new_cost = cost_from(new_width, new_height, new_overlap);
	                    if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
	                        continue;
	                    }
	                    if (new_cost + 1e-15 < best_new_cost) {
	                        found = true;
	                        best_new_cost = new_cost;
	                        best_new_side = new_side;
                            best_new_min_dim = new_min_dim;
	                        best_new_overlap = new_overlap;
	                        best_cand = cand;
	                        best_poly = std::move(cand_poly);
	                        best_bb = cand_bb;
	                        best_extents = e2;
	                    }
	                }

	                if (!found) {
	                    add_reward(op, false, false, false, curr_cost, curr_cost);
	                    maybe_update_controller(t);
	                    continue;
	                }

	                reward_old_cost = old_cost;
	                reward_new_cost = best_new_cost;
		                accepted = accept_move(old_cost, best_new_cost, T);
		                if (accepted) {
		                    improved_curr = (best_new_cost + 1e-15 < old_cost);
		                    const bool valid = (best_new_overlap <= p.overlap_eps_area);
		                    improved_best = valid && better_than_best(best_new_side, best_new_min_dim);

                            grid.update_position(i, best_cand.x, best_cand.y);
		                    poses[static_cast<size_t>(i)] = best_cand;
		                    polys[static_cast<size_t>(i)] = std::move(best_poly);
		                    bbs[static_cast<size_t>(i)] = best_bb;
	                    gmnx = best_extents.min_x;
	                    gmxx = best_extents.max_x;
	                    gmny = best_extents.min_y;
	                    gmxy = best_extents.max_y;
	                    curr_side = best_new_side;
	                    curr_overlap = best_new_overlap;
	                    curr_cost = best_new_cost;

	                    if (improved_best) {
	                        best.best_side = curr_side;
                            best_min_dim = best_new_min_dim;
	                        best.best_poses = poses;
	                    }
	                } else {
	                    bbs[static_cast<size_t>(i)] = old_bb;
	                    gmnx = old_gmnx;
	                    gmxx = old_gmxx;
	                    gmny = old_gmny;
	                    gmxy = old_gmxy;
	                    curr_side = old_side;
	                    curr_overlap = old_overlap;
	                    curr_cost = old_cost;
	                }
		            