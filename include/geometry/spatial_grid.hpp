#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "geometry/geom.hpp"

class UniformGridIndex {
public:
    UniformGridIndex() = default;

    UniformGridIndex(int n,
                     double cell_size,
                     double min_coord = -100.0,
                     double max_coord = 100.0) {
        reset(n, cell_size, min_coord, max_coord);
    }

    void reset(int n,
               double cell_size,
               double min_coord = -100.0,
               double max_coord = 100.0) {
        n_ = std::max(0, n);
        min_coord_ = min_coord;
        max_coord_ = max_coord;

        cell_size_ = std::max(1e-12, cell_size);
        inv_cell_size_ = 1.0 / cell_size_;

        const double span = max_coord_ - min_coord_;
        nx_ = static_cast<int>(std::ceil(span * inv_cell_size_)) + 1;
        ny_ = nx_;
        if (nx_ < 1) {
            nx_ = 1;
        }
        if (ny_ < 1) {
            ny_ = 1;
        }

        cells_.assign(static_cast<size_t>(nx_ * ny_), {});
        cell_of_.assign(static_cast<size_t>(n_), -1);
        pos_in_cell_.assign(static_cast<size_t>(n_), -1);
    }

    void clear() {
        for (auto& v : cells_) {
            v.clear();
        }
        std::fill(cell_of_.begin(), cell_of_.end(), -1);
        std::fill(pos_in_cell_.begin(), pos_in_cell_.end(), -1);
    }

    void rebuild(const std::vector<TreePose>& poses) {
        clear();
        for (int i = 0; i < static_cast<int>(poses.size()); ++i) {
            insert(i, poses[static_cast<size_t>(i)].x, poses[static_cast<size_t>(i)].y);
        }
    }

    void insert(int idx, double x, double y) {
        const int cid = cell_id(x, y);
        auto& v = cells_[static_cast<size_t>(cid)];
        cell_of_[static_cast<size_t>(idx)] = cid;
        pos_in_cell_[static_cast<size_t>(idx)] = static_cast<int>(v.size());
        v.push_back(idx);
    }

    void erase(int idx) {
        const int cid = cell_of_[static_cast<size_t>(idx)];
        if (cid < 0) {
            return;
        }
        auto& v = cells_[static_cast<size_t>(cid)];
        const int pos = pos_in_cell_[static_cast<size_t>(idx)];
        const int last = v.back();
        v[static_cast<size_t>(pos)] = last;
        v.pop_back();
        pos_in_cell_[static_cast<size_t>(last)] = pos;
        cell_of_[static_cast<size_t>(idx)] = -1;
        pos_in_cell_[static_cast<size_t>(idx)] = -1;
    }

    void update_position(int idx, double x, double y) {
        const int new_cid = cell_id(x, y);
        const int old_cid = cell_of_[static_cast<size_t>(idx)];
        if (old_cid == new_cid) {
            return;
        }
        if (old_cid >= 0) {
            erase(idx);
        }
        insert(idx, x, y);
    }

    void gather(double x, double y, std::vector<int>& out) const {
        out.clear();
        const int cx = cell_coord(x);
        const int cy = cell_coord(y);
        for (int dx = -1; dx <= 1; ++dx) {
            int ix = cx + dx;
            if (ix < 0 || ix >= nx_) {
                continue;
            }
            for (int dy = -1; dy <= 1; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= ny_) {
                    continue;
                }
                const int cid = ix + iy * nx_;
                const auto& v = cells_[static_cast<size_t>(cid)];
                out.insert(out.end(), v.begin(), v.end());
            }
        }
    }

private:
    int cell_coord(double v) const {
        double f = (v - min_coord_) * inv_cell_size_;
        int c = static_cast<int>(std::floor(f));
        if (c < 0) {
            c = 0;
        } else if (c >= nx_) {
            c = nx_ - 1;
        }
        return c;
    }

    int cell_id(double x, double y) const {
        const int cx = cell_coord(x);
        const int cy = cell_coord(y);
        return cx + cy * nx_;
    }

    int n_ = 0;
    double min_coord_ = -100.0;
    double max_coord_ = 100.0;
    double cell_size_ = 1.0;
    double inv_cell_size_ = 1.0;
    int nx_ = 1;
    int ny_ = 1;
    std::vector<std::vector<int>> cells_;
    std::vector<int> cell_of_;
    std::vector<int> pos_in_cell_;
};

