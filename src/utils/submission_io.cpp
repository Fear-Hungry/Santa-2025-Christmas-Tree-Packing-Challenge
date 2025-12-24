#include "utils/submission_io.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "utils/wrap_utils.hpp"

namespace {

constexpr std::array<double, 19> kPow10 = {
    1e0,
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
    1e7,
    1e8,
    1e9,
    1e10,
    1e11,
    1e12,
    1e13,
    1e14,
    1e15,
    1e16,
    1e17,
    1e18,
};

double pow10_clamped(int decimals) {
    if (decimals <= 0) {
        return 1.0;
    }
    const int d = std::min(decimals, static_cast<int>(kPow10.size()) - 1);
    return kPow10[static_cast<size_t>(d)];
}

}  // namespace

bool parse_submission_line(const std::string& line,
                           std::string& id,
                           std::string& sx,
                           std::string& sy,
                           std::string& sdeg) {
    std::stringstream ss(line);
    if (!std::getline(ss, id, ',')) {
        return false;
    }
    if (!std::getline(ss, sx, ',')) {
        return false;
    }
    if (!std::getline(ss, sy, ',')) {
        return false;
    }
    if (!std::getline(ss, sdeg, ',')) {
        return false;
    }
    return true;
}

double parse_prefixed_value(const std::string& s) {
    if (s.empty() || s[0] != 's') {
        throw std::runtime_error("Valor sem prefixo 's': " + s);
    }
    return std::stod(s.substr(1));
}

std::string fmt_submission_id(int n, int idx) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << n << "_" << idx;
    return oss.str();
}

SubmissionPoses load_submission_poses(const std::string& path, int n_max) {
    if (n_max <= 0) {
        throw std::runtime_error("load_submission_poses: n_max precisa ser > 0.");
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Erro ao abrir arquivo: " + path);
    }

    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("Arquivo vazio: " + path);
    }

    SubmissionPoses sub;
    sub.path = path;
    sub.by_n.resize(static_cast<size_t>(n_max + 1));

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::string id, sx, sy, sdeg;
        if (!parse_submission_line(line, id, sx, sy, sdeg)) {
            throw std::runtime_error("Linha inválida em " + path + ": " + line);
        }
        if (id.size() < 5 || id[3] != '_') {
            throw std::runtime_error("Formato de id inválido em " + path + ": " + id);
        }

        int n = std::stoi(id.substr(0, 3));
        int idx = std::stoi(id.substr(4));
        if (n < 1 || n > n_max) {
            throw std::runtime_error("n fora de [1," + std::to_string(n_max) +
                                     "] em " + path + ": " + id);
        }
        if (idx < 0 || idx >= n) {
            throw std::runtime_error("idx fora de [0,n) em " + path + ": " + id);
        }

        const double x = parse_prefixed_value(sx);
        const double y = parse_prefixed_value(sy);
        const double deg = parse_prefixed_value(sdeg);

        if (x < -100.0 || x > 100.0 || y < -100.0 || y > 100.0) {
            throw std::runtime_error("Coordenada fora de [-100,100] em " + path + ": " + id);
        }

        if (sub.by_n[static_cast<size_t>(n)].empty()) {
            std::vector<TreePose> inst;
            inst.assign(static_cast<size_t>(n),
                        TreePose{std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN()});
            sub.by_n[static_cast<size_t>(n)] = std::move(inst);
        } else if (static_cast<int>(sub.by_n[static_cast<size_t>(n)].size()) != n) {
            throw std::runtime_error("Tamanho inconsistente para n=" + std::to_string(n) +
                                     " em " + path);
        }

        sub.by_n[static_cast<size_t>(n)][static_cast<size_t>(idx)] = TreePose{x, y, deg};
    }

    for (int n = 1; n <= n_max; ++n) {
        if (static_cast<int>(sub.by_n[static_cast<size_t>(n)].size()) != n) {
            throw std::runtime_error("Instância n=" + std::to_string(n) +
                                     " ausente/incompleta em " + path);
        }
        for (int i = 0; i < n; ++i) {
            const auto& pose = sub.by_n[static_cast<size_t>(n)][static_cast<size_t>(i)];
            if (!std::isfinite(pose.x) || !std::isfinite(pose.y) || !std::isfinite(pose.deg)) {
                throw std::runtime_error("Faltou linha para id " + fmt_submission_id(n, i) +
                                         " em " + path);
            }
        }
    }

    return sub;
}

double quantize_value(double x, int decimals) {
    if (!std::isfinite(x)) {
        return x;
    }
    if (decimals <= 0) {
        return std::nearbyint(x);
    }
    const double scale = pow10_clamped(decimals);
    return std::nearbyint(x * scale) / scale;
}

TreePose quantize_pose(const TreePose& pose, int decimals) {
    return TreePose{quantize_value(pose.x, decimals),
                    quantize_value(pose.y, decimals),
                    quantize_value(pose.deg, decimals)};
}

TreePose quantize_pose_wrap_deg(const TreePose& pose, int decimals) {
    return TreePose{quantize_value(pose.x, decimals),
                    quantize_value(pose.y, decimals),
                    quantize_value(wrap_deg(pose.deg), decimals)};
}

std::vector<TreePose> quantize_poses(const std::vector<TreePose>& poses, int decimals) {
    std::vector<TreePose> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back(quantize_pose(p, decimals));
    }
    return out;
}

std::vector<TreePose> quantize_poses_wrap_deg(const std::vector<TreePose>& poses, int decimals) {
    std::vector<TreePose> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back(quantize_pose_wrap_deg(p, decimals));
    }
    return out;
}
