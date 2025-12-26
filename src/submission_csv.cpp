#include "santa2025/submission_csv.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "santa2025/constraints.hpp"

namespace santa2025 {
namespace {

constexpr double kClampTol = 1e-9;

std::string trim_copy(std::string_view s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return std::string(s.substr(b, e - b));
}

// Splits into exactly 4 fields (id,x,y,deg). Returns empty vector on failure.
std::vector<std::string> split_csv4(const std::string& line) {
    std::vector<std::string> out;
    out.reserve(4);

    size_t start = 0;
    for (int k = 0; k < 3; ++k) {
        const size_t pos = line.find(',', start);
        if (pos == std::string::npos) {
            return {};
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    out.push_back(line.substr(start));
    return out;
}

std::pair<int, int> parse_id(std::string_view id_raw) {
    const std::string id = trim_copy(id_raw);
    if (id.empty()) {
        throw std::runtime_error("empty id");
    }

    size_t sep = id.find('_');
    if (sep == std::string::npos) {
        sep = id.find('.');
    }
    if (sep == std::string::npos) {
        throw std::runtime_error("invalid id (expected NNN_idx): " + id);
    }

    const int puzzle = std::stoi(id.substr(0, sep));
    const int index = std::stoi(id.substr(sep + 1));
    return {puzzle, index};
}

double normalize_deg_360(double deg) {
    double v = std::fmod(deg, 360.0);
    if (v < 0.0) {
        v += 360.0;
    }
    if (std::abs(v) == 0.0) {
        v = 0.0;
    }
    return v;
}

double quantize_for_output(double v, int precision) {
    if (precision <= 0) {
        const double q = std::round(v);
        return (std::abs(q) == 0.0) ? 0.0 : q;
    }
    long double scale = 1.0L;
    for (int i = 0; i < precision; ++i) {
        scale *= 10.0L;
    }
    const long double q = std::round(static_cast<long double>(v) * scale) / scale;
    const double out = static_cast<double>(q);
    return (std::abs(out) == 0.0) ? 0.0 : out;
}

Pose sanitize_pose_for_output(Pose p, int precision) {
    p.x = quantize_for_output(p.x, precision);
    p.y = quantize_for_output(p.y, precision);
    p.deg = quantize_for_output(normalize_deg_360(p.deg), precision);

    if (p.x < kCoordMin - kClampTol || p.x > kCoordMax + kClampTol || p.y < kCoordMin - kClampTol ||
        p.y > kCoordMax + kClampTol) {
        throw std::runtime_error("pose out of bounds after quantize (|delta|>tol) while writing submission");
    }
    p.x = std::clamp(p.x, kCoordMin, kCoordMax);
    p.y = std::clamp(p.y, kCoordMin, kCoordMax);
    return p;
}

}  // namespace

std::string make_submission_id(int puzzle, int index) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << puzzle << "_" << index;
    return oss.str();
}

std::string format_submission_value(double v, int precision) {
    std::ostringstream oss;
    oss << 's' << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

double parse_submission_value(std::string_view token) {
    const std::string s = trim_copy(token);
    if (s.empty()) {
        throw std::runtime_error("empty value");
    }
    if (s[0] != 's' && s[0] != 'S') {
        throw std::runtime_error("expected value with 's' prefix, got: " + s);
    }
    return std::stod(s.substr(1));
}

void write_prefix_submission_csv(std::ostream& out, const std::vector<Pose>& poses, int nmax, int precision) {
    if (nmax <= 0 || nmax > 200) {
        throw std::invalid_argument("write_prefix_submission_csv: nmax must be in [1,200]");
    }
    if (static_cast<int>(poses.size()) < nmax) {
        throw std::invalid_argument("write_prefix_submission_csv: poses.size() must be >= nmax");
    }
    if (precision < 0 || precision > 17) {
        throw std::invalid_argument("write_prefix_submission_csv: precision must be in [0,17]");
    }

    out << "id,x,y,deg\n";
    for (int puzzle = 1; puzzle <= nmax; ++puzzle) {
        for (int i = 0; i < puzzle; ++i) {
            const Pose p = sanitize_pose_for_output(poses[static_cast<size_t>(i)], precision);
            out << make_submission_id(puzzle, i) << "," << format_submission_value(p.x, precision) << ","
                << format_submission_value(p.y, precision) << "," << format_submission_value(p.deg, precision) << "\n";
        }
    }
}

void write_submission_csv(const Submission& sub, std::ostream& out, int nmax, int precision) {
    if (nmax <= 0 || nmax > 200) {
        throw std::invalid_argument("write_submission_csv: nmax must be in [1,200]");
    }
    if (static_cast<int>(sub.poses.size()) < nmax + 1) {
        throw std::invalid_argument("write_submission_csv: sub.poses.size() must be >= nmax+1");
    }
    if (precision < 0 || precision > 17) {
        throw std::invalid_argument("write_submission_csv: precision must be in [0,17]");
    }

    out << "id,x,y,deg\n";
    for (int puzzle = 1; puzzle <= nmax; ++puzzle) {
        const auto& poses = sub.poses[static_cast<size_t>(puzzle)];
        if (static_cast<int>(poses.size()) != puzzle) {
            throw std::invalid_argument("write_submission_csv: poses[puzzle] must have size puzzle");
        }
        for (int i = 0; i < puzzle; ++i) {
            const Pose p = sanitize_pose_for_output(poses[static_cast<size_t>(i)], precision);
            out << make_submission_id(puzzle, i) << "," << format_submission_value(p.x, precision) << ","
                << format_submission_value(p.y, precision) << "," << format_submission_value(p.deg, precision) << "\n";
        }
    }
}

Submission read_submission_csv(std::istream& in, const ReadSubmissionOptions& opt) {
    if (opt.nmax <= 0 || opt.nmax > 200) {
        throw std::invalid_argument("read_submission_csv: nmax must be in [1,200]");
    }

    Submission sub;
    sub.poses.resize(static_cast<size_t>(opt.nmax + 1));
    std::vector<std::vector<bool>> seen(static_cast<size_t>(opt.nmax + 1));

    std::string line;
    int line_no = 0;

    while (std::getline(in, line)) {
        line_no++;
        if (line_no == 1) {
            // Header.
            const std::string hdr = trim_copy(line);
            if (!hdr.empty() && (hdr.rfind("id,", 0) == 0 || hdr == "id")) {
                continue;
            }
            // No header: fall through and parse as data.
        }
        if (trim_copy(line).empty()) {
            continue;
        }

        const auto fields = split_csv4(line);
        if (fields.size() != 4) {
            throw std::runtime_error("line " + std::to_string(line_no) + ": expected 4 columns");
        }

        const auto [puzzle, index] = parse_id(fields[0]);
        if (puzzle <= 0 || puzzle > opt.nmax) {
            throw std::runtime_error("line " + std::to_string(line_no) + ": puzzle out of range: " +
                                     std::to_string(puzzle));
        }
        if (index < 0 || index >= puzzle) {
            throw std::runtime_error("line " + std::to_string(line_no) + ": index out of range: " +
                                     std::to_string(index));
        }

        if (sub.poses[static_cast<size_t>(puzzle)].empty()) {
            sub.poses[static_cast<size_t>(puzzle)].assign(static_cast<size_t>(puzzle), Pose{});
            seen[static_cast<size_t>(puzzle)].assign(static_cast<size_t>(puzzle), false);
        }
        if (seen[static_cast<size_t>(puzzle)][static_cast<size_t>(index)]) {
            throw std::runtime_error("line " + std::to_string(line_no) + ": duplicate id");
        }

        Pose p;
        p.x = parse_submission_value(fields[1]);
        p.y = parse_submission_value(fields[2]);
        p.deg = parse_submission_value(fields[3]);

        sub.poses[static_cast<size_t>(puzzle)][static_cast<size_t>(index)] = p;
        seen[static_cast<size_t>(puzzle)][static_cast<size_t>(index)] = true;
    }

    if (opt.require_complete) {
        for (int puzzle = 1; puzzle <= opt.nmax; ++puzzle) {
            if (sub.poses[static_cast<size_t>(puzzle)].empty()) {
                throw std::runtime_error("missing puzzle " + std::to_string(puzzle));
            }
            for (int i = 0; i < puzzle; ++i) {
                if (!seen[static_cast<size_t>(puzzle)][static_cast<size_t>(i)]) {
                    throw std::runtime_error("missing id " + make_submission_id(puzzle, i));
                }
            }
        }
    }

    return sub;
}

}  // namespace santa2025
