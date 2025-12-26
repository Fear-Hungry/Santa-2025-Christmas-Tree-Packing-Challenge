#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "santa2025/geometry.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    double coarse_step = 0.25;
    double fine_step = 0.01;
    std::string out_json;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return std::string(argv[++i]);
        };

        if (a == "--coarse-step") {
            args.coarse_step = std::stod(need("--coarse-step"));
        } else if (a == "--fine-step") {
            args.fine_step = std::stod(need("--fine-step"));
        } else if (a == "--out-json") {
            args.out_json = need("--out-json");
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: tree_info [--coarse-step deg] [--fine-step deg] [--out-json path]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + a);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);

        const santa2025::Polygon poly = santa2025::tree_polygon();
        const double area = santa2025::polygon_area(poly);
        const santa2025::BoundingBox bb0 = santa2025::polygon_bbox(poly);
        const bool sym_y = santa2025::has_reflection_symmetry_y(poly);

        const auto [best_deg, best_side] = santa2025::minimize_bounding_square_rotation(
            poly, 0.0, 90.0, args.coarse_step, args.fine_step
        );
        const double side0 = bb0.square_side();
        const double side45 = santa2025::bounding_square_side_for_rotation(poly, 45.0);

        const double lb_n1 = std::sqrt(1.0 * area);
        const double lb_n200 = std::sqrt(200.0 * area);

        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\n";
        out << "  \"area\": " << area << ",\n";
        out << "  \"bbox_deg0\": {\n";
        out << "    \"min_x\": " << bb0.min_x << ",\n";
        out << "    \"min_y\": " << bb0.min_y << ",\n";
        out << "    \"max_x\": " << bb0.max_x << ",\n";
        out << "    \"max_y\": " << bb0.max_y << ",\n";
        out << "    \"width\": " << bb0.width() << ",\n";
        out << "    \"height\": " << bb0.height() << ",\n";
        out << "    \"square_side\": " << bb0.square_side() << "\n";
        out << "  },\n";
        out << "  \"bounding_square\": {\n";
        out << "    \"deg0_side\": " << side0 << ",\n";
        out << "    \"deg45_side\": " << side45 << ",\n";
        out << "    \"best_deg_in_0_90\": " << best_deg << ",\n";
        out << "    \"best_side_in_0_90\": " << best_side << "\n";
        out << "  },\n";
        out << "  \"lower_bounds\": {\n";
        out << "    \"n1_side_lb\": " << lb_n1 << ",\n";
        out << "    \"n200_side_lb\": " << lb_n200 << "\n";
        out << "  },\n";
        out << "  \"num_vertices\": " << poly.size() << ",\n";
        out << "  \"symmetry\": { \"reflection_y_axis\": " << (sym_y ? "true" : "false") << " }\n";
        out << "}\n";

        const std::string payload = out.str();
        std::cout << payload;
        if (!args.out_json.empty()) {
            std::ofstream f(args.out_json);
            if (!f) {
                throw std::runtime_error("failed to open --out-json file: " + args.out_json);
            }
            f << payload;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
