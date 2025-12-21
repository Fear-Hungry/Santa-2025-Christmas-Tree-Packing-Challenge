
                if (soft_overlap || n <= 1 || p.squeeze_pushes <= 0) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const double old_cost = curr_cost;
                bool moved_any = false;

                auto attempt_push = [&](bool axis_x, bool pick_min_side) -> bool {
                    const double boundary_tol = 1e-9;

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

                    if (candidates.empty()) {
                        return false;
                    }
                    std::uniform_int_distribution<int> pick(0, static_cast<int>(candidates.size()) - 1);
                    i = candidates[static_cast<size_t>(pick(rng))];

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
                        return false;
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
                                return false;
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
                        return false;
                    }

                    const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
                    const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny, old_gmxy = gmxy;
                    const double old_side = curr_side;
                    const double old_cost_local = curr_cost;

                    bbs[static_cast<size_t>(i)] = best_bb;
                    gmnx = std::min(gmnx, best_bb.min_x);
                    gmxx = std::max(gmxx, best_bb.max_x);
                    gmny = std::min(gmny, best_bb.min_y);
                    gmxy = std::max(gmxy, best_bb.max_y);

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
                    double new_cost = cost_from(new_width, new_height, curr_overlap);
                    if (!(new_cost <= old_cost_local + 1e-15)) {
                        bbs[static_cast<size_t>(i)] = old_bb;
                        gmnx = old_gmnx;
                        gmxx = old_gmxx;
                        gmny = old_gmny;
                        gmxy = old_gmxy;
                        curr_side = old_side;
                        curr_cost = old_cost_local;
                        return false;
                    }

                    grid.update_position(i, best_pose.x, best_pose.y);
                    poses[static_cast<size_t>(i)] = best_pose;
                    polys[static_cast<size_t>(i)] = std::move(best_poly);
                    curr_side = new_side;
                    curr_cost = new_cost;
                    return true;
                };

                for (int rep = 0; rep < p.squeeze_pushes; ++rep) {
                    const double width = gmxx - gmnx;
                    const double height = gmxy - gmny;
                    const bool axis_x = (width >= height);
                    const bool pick_min_side = (uni(rng) < 0.5);
                    if (attempt_push(axis_x, pick_min_side)) {
                        moved_any = true;
                    }
                }

                reward_old_cost = old_cost;
                reward_new_cost = curr_cost;
                accepted = moved_any;
                improved_curr = accepted && (curr_cost + 1e-15 < old_cost);
                const double curr_min_dim = std::min(gmxx - gmnx, gmxy - gmny);
                improved_best = accepted && better_than_best(curr_side, curr_min_dim);
                if (improved_best) {
                    best.best_side = curr_side;
                    best_min_dim = curr_min_dim;
                    best.best_poses = poses;
                }
            