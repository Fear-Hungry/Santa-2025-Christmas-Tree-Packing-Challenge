
                if (soft_overlap || n <= 1) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                sa_refine::PickedIndex pick = picker.pick_index(p.p_pick_extreme);
                int i = pick.idx;

                double dir_x = 0.0;
                double dir_y = 0.0;
                int dir_idx = 0;
                if (!pick_slide_direction(i, dir_x, dir_y, dir_idx)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                TreePose cand_pose;
                Polygon cand_poly;
                BoundingBox cand_bb;
                double cand_delta = 0.0;
                if (!slide_contact_search(i,
                                          dir_idx,
                                          dir_x,
                                          dir_y,
                                          cand_pose,
                                          cand_poly,
                                          cand_bb,
                                          cand_delta)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny, old_gmxy = gmxy;
                const double old_side = curr_side;
                const double old_cost = curr_cost;

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
                double new_cost = cost_from(new_width, new_height, curr_overlap);
                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                    new_cost = std::numeric_limits<double>::infinity();
                }

                reward_old_cost = old_cost;
                reward_new_cost = new_cost;
                accepted = (new_cost <= old_cost + 1e-15);
                if (accepted) {
                    improved_curr = (new_cost + 1e-15 < old_cost);
                    improved_best = better_than_best(new_side, new_min_dim);
                    grid.update_position(i, cand_pose.x, cand_pose.y);
                    poses[static_cast<size_t>(i)] = cand_pose;
                    polys[static_cast<size_t>(i)] = std::move(cand_poly);
                    curr_side = new_side;
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
                    curr_cost = old_cost;
                }
            
