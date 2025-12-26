#include "santa2025/packing_stats.hpp"

#include <algorithm>
#include <limits>

namespace santa2025 {

PackingStats packing_stats(const Polygon& tree_poly, const std::vector<Pose>& poses) {
    PackingStats st;
    if (poses.empty()) {
        st.bbox = BoundingBox{};
        st.square_side = 0.0;
        return st;
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto& p : poses) {
        const Polygon rot = rotate_polygon(tree_poly, p.deg);
        const BoundingBox bb = polygon_bbox(translate_polygon(rot, p.x, p.y));
        min_x = std::min(min_x, bb.min_x);
        min_y = std::min(min_y, bb.min_y);
        max_x = std::max(max_x, bb.max_x);
        max_y = std::max(max_y, bb.max_y);
    }
    st.bbox = BoundingBox{min_x, min_y, max_x, max_y};
    st.square_side = st.bbox.square_side();
    return st;
}

}  // namespace santa2025

