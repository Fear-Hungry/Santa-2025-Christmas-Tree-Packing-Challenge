
                const bool allow_soft_push = soft_overlap && (p.push_overshoot_frac > 0.0);
                if (n <= 1 || (soft_overlap && !allow_soft_push)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const double width = gmxx - gmnx;
                const double height = gmxy - gmny;
                const bool axis_x = (width >= height);
                const double boundary_tol = 1e-9;

                const bool pick_min_side = (uni(rng) < 0.5);
                int i = -1;
                double dir_x = 0.0;
                double dir_y = 0.0;

                std::vector<int> candidates;
                candidates.reserve(static_cast<size_t>(n));
                if (axis_x) {
                    if (pick_min_side) {
                        for (int k = 0; k < n; ++k) {
                            if (!is_active(k)) {
                                continue;
                            }
                            if (bbs[static_cast<size_t>(k)].min_x <= gmnx + boundary_tol) {
                                candidates.push_back(k);
                            }
                        }
                        dir_x = 1.0;
                    } else {
                        for (int k = 0; k < n; ++k) {
                            if (!is_active(k)) {
                                continue;
                            }
                            if (bbs[static_cast<size_t>(k)].max_x >= gmxx - boundary_tol) {
                                candidates.push_back(k);
                            }
                        }
                        dir_x = -1.0;
                    }
                } else {
                    if (pick_min_side) {
                        for (int k = 0; k < n; ++k) {
                            if (!is_active(k)) {
                                continue;
                            }
                            if (bbs[static_cast<size_t>(k)].min_y <= gmny + boundary_tol) {
                                candidates.push_back(k);
                            }
                        }
                        dir_y = 1.0;
                    } else {
                        for (int k = 0; k < n; ++k) {
                            if (!is_active(k)) {
                                continue;
                            }
                            if (bbs[static_cast<size_t>(k)].max_y >= gmxy - boundary_tol) {
                                candidates.push_back(k);
                            }
                        }
                        dir_y = -1.0;
                    }
                }

                if (!candidates.empty()) {
                    std::uniform_int_distribution<int> pick(0,
                                                             static_cast<int>(candidates.size()) - 1);
                    i = candidates[static_cast<size_t>(pick(rng))];
                } else {
                    i = picker.pick_index(p.p_pick_extreme).idx;
                    if (axis_x) {
                        dir_x = (uni(rng) < 0.5) ? 1.0 : -1.0;
                    } else {
                        dir_y = (uni(rng) < 0.5) ? 1.0 : -1.0;
                    }
                }

                double max_step =
                    p.push_max_step_frac * std::max(1e-9, curr_side);
                if (axis_x) {
                    if (dir_x > 0.0) {
                        max_step = std::min(max_step, 100.0 - poses[static_cast<size_t>(i)].x);
                    } else {
                        max_step = std::min(max_step, poses[static_cast<size_t>(i)].x + 100.0);
                    }
                } else {
                    if (dir_y > 0.0) {
                        max_step = std::min(max_step, 100.0 - poses[static_cast<size_t>(i)].y);
                    } else {
                        max_step = std::min(max_step, poses[static_cast<size_t>(i)].y + 100.0);
                    }
                }
                max_step = std::min(max_step, thr);
                if (!(max_step > 1e-12)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

	                auto valid_at = [&](double delta,
	                                    TreePose& pose_out,
	                                    Polygon& poly_out,
	                                    BoundingBox& bb_out) -> bool {
	                    pose_out = poses[static_cast<size_t>(i)];
	                    pose_out.x += dir_x * delta;
	                    pose_out.y += dir_y * delta;
	                    quantize_pose_inplace(pose_out);
	                    if (pose_out.x < -100.0 || pose_out.x > 100.0 || pose_out.y < -100.0 ||
	                        pose_out.y > 100.0) {
	                        return false;
	                    }
	                    poly_out = transform_polygon(base_poly_, pose_out);
                    bb_out = bounding_box(poly_out);

                    grid.gather(pose_out.x, pose_out.y, neigh);
                    for (int j : neigh) {
                        if (j == i) {
                            continue;
                        }
                        double ddx = pose_out.x - poses[static_cast<size_t>(j)].x;
                        double ddy = pose_out.y - poses[static_cast<size_t>(j)].y;
                        if (ddx * ddx + ddy * ddy > thr_sq) {
                            continue;
                        }
                        if (!aabb_overlap(bb_out, bbs[static_cast<size_t>(j)])) {
                            continue;
                        }
                        if (polygons_intersect(poly_out, polys[static_cast<size_t>(j)])) {
                            return false;
                        }
                    }
                    return true;
                };

                TreePose best_pose;
                Polygon best_poly;
                BoundingBox best_bb;
                double best_delta = 0.0;
                {
                    TreePose pose_hi;
                    Polygon poly_hi;
                    BoundingBox bb_hi;
                    if (valid_at(max_step, pose_hi, poly_hi, bb_hi)) {
                        best_pose = pose_hi;
                        best_poly = std::move(poly_hi);
                        best_bb = bb_hi;
                        best_delta = max_step;
                    } else {
                        double hi_invalid = max_step;
                        double lo_valid = 0.0;
                        TreePose pose_lo;
                        Polygon poly_lo;
                        BoundingBox bb_lo;
                        bool found = false;
                        double step = max_step;
                        for (int bt = 0; bt < 14; ++bt) {
                            step *= 0.5;
                            if (!(step > 1e-12)) {
                                break;
                            }
                            if (valid_at(step, pose_lo, poly_lo, bb_lo)) {
                                lo_valid = step;
                                found = true;
                                break;
                            }
                            hi_invalid = step;
                        }
                        if (!found) {
                            add_reward(op, false, false, false, curr_cost, curr_cost);
                            maybe_update_controller(t);
                            continue;
                        }

                        double lo = lo_valid;
                        double hi = hi_invalid;
                        best_pose = pose_lo;
                        best_poly = std::move(poly_lo);
                        best_bb = bb_lo;
                        best_delta = lo;

                        for (int it = 0; it < p.push_bisect_iters; ++it) {
                            double mid = 0.5 * (lo + hi);
                            TreePose pose_mid;
                            Polygon poly_mid;
                            BoundingBox bb_mid;
                            if (valid_at(mid, pose_mid, poly_mid, bb_mid)) {
                                lo = mid;
                                best_pose = pose_mid;
                                best_poly = std::move(poly_mid);
                                best_bb = bb_mid;
                                best_delta = lo;
                            } else {
                                hi = mid;
                            }
                        }
                    }
                }

                if (!(best_delta > 1e-12)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                TreePose cand_pose = best_pose;
                Polygon cand_poly = best_poly;
                BoundingBox cand_bb = best_bb;
                double cand_delta = best_delta;
                if (soft_overlap) {
                    const double overshoot = p.push_overshoot_frac * max_step;
                    if (overshoot > 0.0) {
                        double target_delta = std::min(max_step, best_delta + overshoot);
                        if (target_delta > best_delta + 1e-12) {
                            double delta = target_delta;
                            bool ok = false;
                            for (int bt = 0; bt < 6; ++bt) {
                                TreePose pose_out = poses[static_cast<size_t>(i)];
                                pose_out.x += dir_x * delta;
                                pose_out.y += dir_y * delta;
                                quantize_pose_inplace(pose_out);
                                if (pose_out.x < -100.0 || pose_out.x > 100.0 ||
                                    pose_out.y < -100.0 || pose_out.y > 100.0) {
                                    delta = best_delta + 0.5 * (delta - best_delta);
                                    continue;
                                }
                                Polygon poly_out = transform_polygon(base_poly_, pose_out);
                                BoundingBox bb_out = bounding_box(poly_out);
                                cand_pose = pose_out;
                                cand_poly = std::move(poly_out);
                                cand_bb = bb_out;
                                cand_delta = delta;
                                ok = true;
                                break;
                            }
                            if (!ok) {
                                cand_pose = best_pose;
                                cand_poly = best_poly;
                                cand_bb = best_bb;
                                cand_delta = best_delta;
                            }
                        }
                    }
                }

                if (!(cand_delta > 1e-12)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny, old_gmxy = gmxy;
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
                double new_overlap = curr_overlap;
                if (soft_overlap) {
                    double delta_overlap = 0.0;
                    gather_union(i,
                                 -1,
                                 poses[static_cast<size_t>(i)].x,
                                 poses[static_cast<size_t>(i)].y,
                                 cand_pose.x,
                                 cand_pose.y,
                                 neigh_union);
                    for (int j : neigh_union) {
                        const double old_dx =
                            poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(j)].x;
                        const double old_dy =
                            poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(j)].y;
                        const double new_dx = cand_pose.x - poses[static_cast<size_t>(j)].x;
                        const double new_dy = cand_pose.y - poses[static_cast<size_t>(j)].y;

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
                            new_a = overlap_metric(cand_pose,
                                                   poses[static_cast<size_t>(j)]);
                        }
                        delta_overlap += clamp_overlap(new_a) - clamp_overlap(old_a);
                    }
                    new_overlap = std::max(0.0, old_overlap + delta_overlap);
                    new_overlap = clamp_overlap(new_overlap);
                }

                double new_cost = cost_from(new_width, new_height, new_overlap);
                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                    new_cost = std::numeric_limits<double>::infinity();
                }

                reward_old_cost = old_cost;
                reward_new_cost = new_cost;
                if (soft_overlap) {
                    accepted = accept_move(old_cost, new_cost, T);
                } else {
                    accepted = (new_cost <= old_cost + 1e-15);
                }
                if (accepted) {
                    improved_curr = (new_cost + 1e-15 < old_cost);
                    const bool valid =
                        soft_overlap ? (new_overlap <= p.overlap_eps_area) : true;
                    improved_best = valid && better_than_best(new_side, new_min_dim);
                    grid.update_position(i, cand_pose.x, cand_pose.y);
                    poses[static_cast<size_t>(i)] = cand_pose;
                    polys[static_cast<size_t>(i)] = std::move(cand_poly);
                    curr_side = new_side;
                    if (soft_overlap) {
                        curr_overlap = new_overlap;
                    }
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
                    if (soft_overlap) {
                        curr_overlap = old_overlap;
                    }
                    curr_cost = old_cost;
                }
            