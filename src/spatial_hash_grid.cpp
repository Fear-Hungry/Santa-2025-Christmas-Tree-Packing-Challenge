#include "santa2025/spatial_hash_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace santa2025 {

SpatialHashGrid::SpatialHashGrid(double cell_size, double object_radius)
    : cell_size_(cell_size), object_radius_(object_radius) {
    if (!(cell_size_ > 0.0)) {
        throw std::invalid_argument("SpatialHashGrid: cell_size must be > 0");
    }
    if (!(object_radius_ >= 0.0)) {
        throw std::invalid_argument("SpatialHashGrid: object_radius must be >= 0");
    }
}

void SpatialHashGrid::clear() {
    cells_.clear();
    for (auto& v : id_cells_) {
        v.clear();
    }
    std::fill(id_present_.begin(), id_present_.end(), false);
    std::fill(seen_.begin(), seen_.end(), 0);
    stamp_ = 1;
}

void SpatialHashGrid::reserve_ids(size_t n) {
    id_cells_.resize(n);
    id_present_.assign(n, false);
    seen_.assign(n, 0);
}

std::int64_t SpatialHashGrid::coord_to_cell(double v, double cell_size) {
    return static_cast<std::int64_t>(std::floor(v / cell_size));
}

std::vector<SpatialHashGrid::CellKey> SpatialHashGrid::cells_for_circle(double x, double y, double r) const {
    const double rr = (r >= 0.0) ? r : 0.0;
    const std::int64_t ix0 = coord_to_cell(x - rr, cell_size_);
    const std::int64_t ix1 = coord_to_cell(x + rr, cell_size_);
    const std::int64_t iy0 = coord_to_cell(y - rr, cell_size_);
    const std::int64_t iy1 = coord_to_cell(y + rr, cell_size_);

    std::vector<CellKey> keys;
    keys.reserve(static_cast<size_t>((ix1 - ix0 + 1) * (iy1 - iy0 + 1)));
    for (std::int64_t ix = ix0; ix <= ix1; ++ix) {
        for (std::int64_t iy = iy0; iy <= iy1; ++iy) {
            keys.push_back(CellKey{ix, iy});
        }
    }
    return keys;
}

void SpatialHashGrid::insert(int id, double x, double y) {
    if (id < 0) {
        throw std::invalid_argument("SpatialHashGrid::insert: id must be >= 0");
    }
    const size_t uid = static_cast<size_t>(id);
    if (uid >= id_cells_.size()) {
        reserve_ids(uid + 1);
    }
    if (id_present_[uid]) {
        throw std::runtime_error("SpatialHashGrid::insert: id already present");
    }

    auto keys = cells_for_circle(x, y, object_radius_);
    id_cells_[uid] = keys;
    id_present_[uid] = true;

    for (const auto& k : keys) {
        cells_[k].push_back(id);
    }
}

void SpatialHashGrid::remove(int id) {
    if (id < 0) {
        return;
    }
    const size_t uid = static_cast<size_t>(id);
    if (uid >= id_cells_.size() || !id_present_[uid]) {
        return;
    }

    for (const auto& k : id_cells_[uid]) {
        auto it = cells_.find(k);
        if (it == cells_.end()) {
            continue;
        }
        auto& vec = it->second;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] == id) {
                vec[i] = vec.back();
                vec.pop_back();
                break;
            }
        }
        if (vec.empty()) {
            cells_.erase(it);
        }
    }

    id_cells_[uid].clear();
    id_present_[uid] = false;
}

void SpatialHashGrid::update(int id, double x, double y) {
    if (id < 0) {
        throw std::invalid_argument("SpatialHashGrid::update: id must be >= 0");
    }
    const size_t uid = static_cast<size_t>(id);
    if (uid >= id_cells_.size()) {
        reserve_ids(uid + 1);
    }
    if (id_present_[uid]) {
        remove(id);
    }
    insert(id, x, y);
}

void SpatialHashGrid::query_into(std::vector<int>& out, double x, double y, double radius) const {
    out.clear();

    // Prevent stamp overflow from breaking dedupe.
    if (++stamp_ == 0) {
        std::fill(seen_.begin(), seen_.end(), 0);
        stamp_ = 1;
    }

    auto keys = cells_for_circle(x, y, radius);
    for (const auto& k : keys) {
        auto it = cells_.find(k);
        if (it == cells_.end()) {
            continue;
        }
        for (const int id : it->second) {
            if (id < 0) {
                continue;
            }
            const size_t uid = static_cast<size_t>(id);
            if (uid >= seen_.size()) {
                continue;
            }
            if (seen_[uid] == stamp_) {
                continue;
            }
            seen_[uid] = stamp_;
            out.push_back(id);
        }
    }
}

}  // namespace santa2025

