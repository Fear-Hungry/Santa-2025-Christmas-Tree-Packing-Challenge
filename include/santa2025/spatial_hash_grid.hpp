#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace santa2025 {

class SpatialHashGrid {
public:
    struct CellKey {
        std::int64_t ix = 0;
        std::int64_t iy = 0;

        bool operator==(const CellKey& other) const { return ix == other.ix && iy == other.iy; }
    };

    struct CellKeyHash {
        size_t operator()(const CellKey& k) const noexcept {
            // 64-bit mix (good enough for small n; keeps it dependency-free).
            const std::uint64_t x = static_cast<std::uint64_t>(k.ix);
            const std::uint64_t y = static_cast<std::uint64_t>(k.iy);
            std::uint64_t h = x * 0x9E3779B97F4A7C15ULL;
            h ^= y + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
            return static_cast<size_t>(h);
        }
    };

    SpatialHashGrid(double cell_size, double object_radius);

    void clear();
    void reserve_ids(size_t n);

    void insert(int id, double x, double y);
    void remove(int id);
    void update(int id, double x, double y);

    // Returns candidate IDs whose occupied cells overlap the query circle.
    // Deduplicates IDs across multiple cells.
    void query_into(std::vector<int>& out, double x, double y, double radius) const;

    double cell_size() const { return cell_size_; }
    double object_radius() const { return object_radius_; }

private:
    double cell_size_ = 1.0;
    double object_radius_ = 0.0;

    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> cells_;
    std::vector<std::vector<CellKey>> id_cells_;
    std::vector<bool> id_present_;

    mutable std::vector<std::uint32_t> seen_;
    mutable std::uint32_t stamp_ = 1;

    std::vector<CellKey> cells_for_circle(double x, double y, double r) const;
    static std::int64_t coord_to_cell(double v, double cell_size);
};

}  // namespace santa2025

