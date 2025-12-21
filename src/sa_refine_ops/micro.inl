
	                sa_refine::PickedIndex pick = picker.pick_index(p.p_pick_extreme);
	                int i = pick.idx;

                double vxdir = cx - poses[static_cast<size_t>(i)].x;
                double vydir = cy - poses[static_cast<size_t>(i)].y;
                double vnorm = std::hypot(vxdir, vydir) + 1e-12;
                vxdir /= vnorm;
                vydir /= vnorm;

                double dx = 0.0;
                double dy = 0.0;
                if (uni(rng) < p.p_random_dir) {
                    dx = normal(rng) * step;
                    dy = normal(rng) * step;
                } else {
                    double mag = std::abs(normal(rng)) * step;
                    bool used_boundary_dir = false;
                    if (pick.extreme) {
                        const double width = gmxx - gmnx;
                        const double height = gmxy - gmny;
                        const double boundary_tol = 1e-9;
                        const auto& bb = bbs[static_cast<size_t>(i)];
                        if (width >= height) {
                            bool left = (bb.min_x <= gmnx + boundary_tol);
                            bool right = (bb.max_x >= gmxx - boundary_tol);
                            if (left || right) {
                                used_boundary_dir = true;
                                double dir = left ? 1.0 : -1.0;
                                dx = dir * mag + normal(rng) * (0.10 * step);
                                dy = normal(rng) * (0.35 * step);
                            }
                        } else {
                            bool bottom = (bb.min_y <= gmny + boundary_tol);
                            bool top = (bb.max_y >= gmxy - boundary_tol);
                            if (bottom || top) {
                                used_boundary_dir = true;
                                double dir = bottom ? 1.0 : -1.0;
                                dx = normal(rng) * (0.35 * step);
                                dy = dir * mag + normal(rng) * (0.10 * step);
                            }
                        }
                    }
                    if (!used_boundary_dir) {
                        dx = vxdir * mag + normal(rng) * (0.25 * step);
                        dy = vydir * mag + normal(rng) * (0.25 * step);
                    }
                }

	                if (p.kick_prob > 0.0 && uni(rng) < p.kick_prob) {
	                    dx *= p.kick_mult;
	                    dy *= p.kick_mult;
	                }

	                const bool do_rot = (uni(rng) < p.p_rot);
	                const double ddeg = do_rot ? (uni_deg(rng) * ddeg_rng) : 0.0;

	                TreePose cand;
	                Polygon cand_poly;
	                BoundingBox cand_bb;
	                bool ok = false;
	                double scale = 1.0;
		                for (int bt = 0; bt < 6; ++bt) {
		                    cand = poses[static_cast<size_t>(i)];
		                    cand.x += dx * scale;
		                    cand.y += dy * scale;
		                    if (do_rot) {
		                        cand.deg = wrap_deg(cand.deg + ddeg);
		                    }
		                    quantize_pose_inplace(cand);

		                    if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 ||
		                        cand.y > 100.0) {
		                        scale *= 0.5;
		                        continue;
		                    }

	                    cand_poly = transform_polygon(base_poly_, cand);
	                    cand_bb = bounding_box(cand_poly);

		                    if (!soft_overlap) {
		                        ok = true;
                                grid.gather(cand.x, cand.y, neigh);
                                for (int j : neigh) {
                                    if (j == i) {
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
                                        ok = false;
                                        break;
                                    }
                                }
		                        if (ok) {
		                            break;
		                        }
		                        scale *= 0.5;
		                        continue;
		                    }
		                    ok = true;
		                    break;
		                }
		                if (!ok) {
		                    add_reward(op, false, false, false, curr_cost, curr_cost);
		                    maybe_update_controller(t);
		                    continue;
		                }

	                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
	                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
	                             old_gmxy = gmxy;
	                const double old_side = curr_side;
	                const double old_cost = curr_cost;
	                const double old_overlap = curr_overlap;

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

                    const double new_width = gmxx - gmnx;
                    const double new_height = gmxy - gmny;
	                double new_side = std::max(new_width, new_height);
                    const double new_min_dim = std::min(new_width, new_height);
	                double delta_overlap = 0.0;
	                if (soft_overlap) {
                        gather_union(i,
                                     -1,
                                     poses[static_cast<size_t>(i)].x,
                                     poses[static_cast<size_t>(i)].y,
                                     cand.x,
                                     cand.y,
                                     neigh_union);
	                    for (int j : neigh_union) {
	                        const double old_dx =
	                            poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(j)].x;
	                        const double old_dy =
	                            poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(j)].y;
	                        const double new_dx = cand.x - poses[static_cast<size_t>(j)].x;
	                        const double new_dy = cand.y - poses[static_cast<size_t>(j)].y;

	                        double old_a = 0.0;
	                        if ((old_dx * old_dx + old_dy * old_dy <= thr_sq) &&
	                            aabb_overlap(old_bb, bbs[static_cast<size_t>(j)]) &&
	                            polygons_intersect(polys[static_cast<size_t>(i)],
	                                               polys[static_cast<size_t>(j)])) {
	                            old_a = overlap_metric(poses[static_cast<size_t>(i)],
	                                                   poses[static_cast<size_t>(j)]);
	                        }
	                        double new_a = 0.0;
	                        if ((new_dx * new_dx + new_dy * new_dy <= thr_sq) &&
	                            aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)]) &&
	                            polygons_intersect(cand_poly,
	                                               polys[static_cast<size_t>(j)])) {
	                            new_a = overlap_metric(cand,
	                                                   poses[static_cast<size_t>(j)]);
	                        }
	                        delta_overlap += clamp_overlap(new_a) - clamp_overlap(old_a);
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
                        grid.update_position(i, cand.x, cand.y);
	                    poses[static_cast<size_t>(i)] = cand;
	                    polys[static_cast<size_t>(i)] = std::move(cand_poly);
	                    curr_side = new_side;
	                    curr_overlap = new_overlap;
	                    curr_cost = new_cost;
	                    if (improved_best) {
	                        best.best_side = curr_side;
                            best_min_dim = new_min_dim;
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
            
