#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "geom.hpp"

namespace sa_refine {

struct UniformGrid {
    double cell_size = 1.0;
    double inv_cell_size = 1.0;
    int nx = 1;
    int ny = 1;
    std::vector<std::vector<int>> cells;
    std::vector<int> cell_of;
    std::vector<int> pos_in_cell;

    UniformGrid(int n, double cell_size_in)
        : cell_size(std::max(1e-12, cell_size_in)),
          inv_cell_size(1.0 / std::max(1e-12, cell_size_in)) {
        const double span = 200.0;
        nx = static_cast<int>(std::ceil(span * inv_cell_size)) + 1;
        ny = nx;
        cells.resize(static_cast<size_t>(nx * ny));
        cell_of.assign(static_cast<size_t>(n), -1);
        pos_in_cell.assign(static_cast<size_t>(n), -1);
    }

    int cell_coord(double v) const {
        double f = (v + 100.0) * inv_cell_size;
        int c = static_cast<int>(std::floor(f));
        if (c < 0) {
            c = 0;
        } else if (c >= nx) {
            c = nx - 1;
        }
        return c;
    }

    int cell_id(double x, double y) const {
        const int cx = cell_coord(x);
        const int cy = cell_coord(y);
        return cx + cy * nx;
    }

    void clear() {
        for (auto& v : cells) {
            v.clear();
        }
        std::fill(cell_of.begin(), cell_of.end(), -1);
        std::fill(pos_in_cell.begin(), pos_in_cell.end(), -1);
    }

    void insert(int idx, double x, double y) {
        const int cid = cell_id(x, y);
        auto& v = cells[static_cast<size_t>(cid)];
        cell_of[static_cast<size_t>(idx)] = cid;
        pos_in_cell[static_cast<size_t>(idx)] = static_cast<int>(v.size());
        v.push_back(idx);
    }

    void erase(int idx) {
        const int cid = cell_of[static_cast<size_t>(idx)];
        if (cid < 0) {
            return;
        }
        auto& v = cells[static_cast<size_t>(cid)];
        const int pos = pos_in_cell[static_cast<size_t>(idx)];
        const int last = v.back();
        v[static_cast<size_t>(pos)] = last;
        v.pop_back();
        pos_in_cell[static_cast<size_t>(last)] = pos;
        cell_of[static_cast<size_t>(idx)] = -1;
        pos_in_cell[static_cast<size_t>(idx)] = -1;
    }

    void update_position(int idx, double x, double y) {
        const int new_cid = cell_id(x, y);
        const int old_cid = cell_of[static_cast<size_t>(idx)];
        if (old_cid == new_cid) {
            return;
        }
        if (old_cid >= 0) {
            erase(idx);
        }
        insert(idx, x, y);
    }

    void rebuild(const std::vector<TreePose>& poses) {
        clear();
        for (int i = 0; i < static_cast<int>(poses.size()); ++i) {
            insert(i,
                   poses[static_cast<size_t>(i)].x,
                   poses[static_cast<size_t>(i)].y);
        }
    }

    void gather(double x, double y, std::vector<int>& out) const {
        out.clear();
        const int cx = cell_coord(x);
        const int cy = cell_coord(y);
        for (int dx = -1; dx <= 1; ++dx) {
            int ix = cx + dx;
            if (ix < 0 || ix >= nx) {
                continue;
            }
            for (int dy = -1; dy <= 1; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= ny) {
                    continue;
                }
                const int cid = ix + iy * nx;
                const auto& v = cells[static_cast<size_t>(cid)];
                out.insert(out.end(), v.begin(), v.end());
            }
        }
    }
};

}  // namespace sa_refine
