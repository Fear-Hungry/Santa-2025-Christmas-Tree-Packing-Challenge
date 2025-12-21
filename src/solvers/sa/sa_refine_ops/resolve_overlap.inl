
                if (!soft_overlap || n <= 1 || curr_overlap <= p.overlap_eps_area) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                bool moved = false;
                double step =
                    (p.resolve_step_frac_max * (1.0 - frac) + p.resolve_step_frac_min * frac) *
                    std::max(1e-9, curr_side);
                step *= step_mult;

                for (int att = 0; att < std::max(1, p.resolve_attempts) && !moved; ++att) {
                    int i = -1;
                    Point rep{0.0, 0.0};

                    for (int tries = 0; tries < 8; ++tries) {
                        int cand_i = picker.pick_index(p.p_pick_extreme).idx;

                        Point best_rep{0.0, 0.0};
                        double best_score = 0.0;
                            grid.gather(poses[static_cast<size_t>(cand_i)].x,
                                        poses[static_cast<size_t>(cand_i)].y,
                                        neigh);
                            for (int j : neigh) {
                                if (j == cand_i) {
                                    continue;
                                }
                                double ddx = poses[static_cast<size_t>(cand_i)].x -
                                             poses[static_cast<size_t>(j)].x;
                                double ddy = poses[static_cast<size_t>(cand_i)].y -
                                             poses[static_cast<size_t>(j)].y;
                                if (ddx * ddx + ddy * ddy > thr_sq) {
                                    continue;
                                }
                                if (!aabb_overlap(bbs[static_cast<size_t>(cand_i)],
                                                  bbs[static_cast<size_t>(j)])) {
                                    continue;
                                }
                                if (!polygons_intersect(polys[static_cast<size_t>(cand_i)],
                                                        polys[static_cast<size_t>(j)])) {
                                    continue;
                                }

                                Point avg_mtv{0.0, 0.0};
                                double area = 0.0;
                                if (!overlap_mtv(poses[static_cast<size_t>(cand_i)],
                                                 poses[static_cast<size_t>(j)],
                                                 avg_mtv,
                                                 area)) {
                                    continue;
                                }
                                double a = clamp_overlap(area);
                                if (!(a > 0.0)) {
                                    continue;
                                }

                                double metric = use_mtv_metric
                                                    ? (avg_mtv.x * avg_mtv.x + avg_mtv.y * avg_mtv.y)
                                                    : a;
                                metric = clamp_overlap(metric);
                                if (metric > best_score) {
                                    best_score = metric;
                                    best_rep = avg_mtv;
                                }
                            }

                        if (best_score > 0.0) {
                            i = cand_i;
                            rep = best_rep;
                            break;
                        }
                    }

                    if (i < 0) {
                        break;
                    }

                    double vnorm = std::hypot(rep.x, rep.y);
                    double dir_x = 0.0;
                    double dir_y = 0.0;
                    double move_len = step;
                    if (vnorm > 1e-12) {
                        dir_x = rep.x / vnorm;
                        dir_y = rep.y / vnorm;
                        move_len = std::min(step, vnorm);
                    } else {
                        double a = uni(rng) * 2.0 * 3.14159265358979323846;
                        dir_x = std::cos(a);
                        dir_y = std::sin(a);
                        move_len = step;
                    }

                    double dx = dir_x * move_len + normal(rng) * (p.resolve_noise_frac * move_len);
                    double dy = dir_y * move_len + normal(rng) * (p.resolve_noise_frac * move_len);

                    double scale = 1.0;
	                    for (int bt = 0; bt < 6 && !moved; ++bt) {
	                        TreePose cand = poses[static_cast<size_t>(i)];
	                        cand.x += dx * scale;
	                        cand.y += dy * scale;
	                        quantize_pose_inplace(cand);
	                        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
	                            scale *= 0.5;
	                            continue;
	                        }

                        Polygon cand_poly = transform_polygon(base_poly_, cand);
                        BoundingBox cand_bb = bounding_box(cand_poly);

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
                                aabb_overlap(bbs[static_cast<size_t>(i)],
                                             bbs[static_cast<size_t>(k)]) &&
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

                        double new_overlap = std::max(0.0, curr_overlap + delta_overlap);
                        new_overlap = clamp_overlap(new_overlap);

                        Extents e2;
                        e2.min_x = std::numeric_limits<double>::infinity();
                        e2.max_x = -std::numeric_limits<double>::infinity();
                        e2.min_y = std::numeric_limits<double>::infinity();
                        e2.max_y = -std::numeric_limits<double>::infinity();
                        for (int k = 0; k < n; ++k) {
                            const BoundingBox& bb =
                                (k == i) ? cand_bb : bbs[static_cast<size_t>(k)];
                            e2.min_x = std::min(e2.min_x, bb.min_x);
                            e2.max_x = std::max(e2.max_x, bb.max_x);
                            e2.min_y = std::min(e2.min_y, bb.min_y);
                            e2.max_y = std::max(e2.max_y, bb.max_y);
                        }
                        const double new_width = e2.max_x - e2.min_x;
                        const double new_height = e2.max_y - e2.min_y;
                        const double new_min_dim = std::min(new_width, new_height);
                        double new_side = std::max(new_width, new_height);
                        double new_cost = cost_from(new_width, new_height, new_overlap);
                        if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                            scale *= 0.5;
                            continue;
                        }

                        reward_old_cost = curr_cost;
                        reward_new_cost = new_cost;
                        bool acc = accept_move(curr_cost, new_cost, T);
                        if (!acc) {
                            scale *= 0.5;
                            continue;
                        }

                        accepted = true;
		                        improved_curr = (new_cost + 1e-15 < curr_cost);
		                        const bool valid = (new_overlap <= p.overlap_eps_area);
		                        improved_best = valid && better_than_best(new_side, new_min_dim);

                            grid.update_position(i, cand.x, cand.y);
	                        poses[static_cast<size_t>(i)] = cand;
	                        polys[static_cast<size_t>(i)] = std::move(cand_poly);
	                        bbs[static_cast<size_t>(i)] = cand_bb;
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
                        moved = true;
                    }
                }
            
