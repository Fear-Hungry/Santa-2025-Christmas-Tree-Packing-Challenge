#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"

namespace {

constexpr int kOutputDecimals = 9;

struct PoseRec {
    std::string sx;
    std::string sy;
    std::string sdeg;
    double x = 0.0;
    double y = 0.0;
    double deg = 0.0;
};

struct SubmissionData {
    std::string path;
    std::vector<std::vector<PoseRec>> by_n;  // 0..200
};

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

std::string fmt_id(int n, int idx) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << n << "_" << idx;
    return oss.str();
}

SubmissionData load_submission(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Erro ao abrir arquivo: " + path);
    }

    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("Arquivo vazio: " + path);
    }

    SubmissionData sub;
    sub.path = path;
    sub.by_n.resize(201);

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
            throw std::runtime_error("Formato de id inválido em " + path +
                                     ": " + id);
        }

        int n = std::stoi(id.substr(0, 3));
        int idx = std::stoi(id.substr(4));
        if (n < 1 || n > 200) {
            throw std::runtime_error("n fora de [1,200] em " + path + ": " + id);
        }
        if (idx < 0 || idx >= n) {
            throw std::runtime_error("idx fora de [0,n) em " + path + ": " + id);
        }

        PoseRec rec;
        rec.sx = sx;
        rec.sy = sy;
        rec.sdeg = sdeg;
        rec.x = parse_prefixed_value(sx);
        rec.y = parse_prefixed_value(sy);
        rec.deg = parse_prefixed_value(sdeg);

        if (rec.x < -100.0 || rec.x > 100.0 || rec.y < -100.0 || rec.y > 100.0) {
            throw std::runtime_error("Coordenada fora de [-100,100] em " + path +
                                     ": " + id);
        }

        if (sub.by_n[static_cast<size_t>(n)].empty()) {
            sub.by_n[static_cast<size_t>(n)].resize(static_cast<size_t>(n));
        } else if (static_cast<int>(sub.by_n[static_cast<size_t>(n)].size()) != n) {
            throw std::runtime_error("Tamanho inconsistente para n=" + std::to_string(n) +
                                     " em " + path);
        }
        sub.by_n[static_cast<size_t>(n)][static_cast<size_t>(idx)] = std::move(rec);
    }

    for (int n = 1; n <= 200; ++n) {
        if (static_cast<int>(sub.by_n[static_cast<size_t>(n)].size()) != n) {
            throw std::runtime_error("Instância n=" + std::to_string(n) +
                                     " ausente/incompleta em " + path);
        }
        for (int i = 0; i < n; ++i) {
            const auto& rec = sub.by_n[static_cast<size_t>(n)][static_cast<size_t>(i)];
            if (rec.sx.empty() || rec.sy.empty() || rec.sdeg.empty()) {
                throw std::runtime_error("Faltou linha para id " + fmt_id(n, i) +
                                         " em " + path);
            }
        }
    }

    return sub;
}

double side_for_solution(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    auto polys = transformed_polygons(base_poly, poses);
    return bounding_square_side(polys);
}

double quantize_value(double x) {
    const std::string s = fmt_submission_value(x, kOutputDecimals);
    return std::stod(s.substr(1));
}

TreePose quantize_pose(const TreePose& pose) {
    return TreePose{quantize_value(pose.x),
                    quantize_value(pose.y),
                    quantize_value(pose.deg)};
}

std::vector<TreePose> quantize_poses(const std::vector<TreePose>& poses) {
    std::vector<TreePose> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back(quantize_pose(p));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Uso: " << argv[0]
                << " output.csv input1.csv [input2.csv ...] [--no-final-rigid]\n";
            return 2;
        }

        const std::string output_path = argv[1];
        bool final_rigid = true;
        std::vector<std::string> inputs;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--no-final-rigid" || arg == "--no-sa-rigid") {
                final_rigid = false;
                continue;
            }
            inputs.push_back(std::move(arg));
        }
        if (inputs.empty()) {
            throw std::runtime_error("Precisa de ao menos 1 input.");
        }

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);

        std::vector<SubmissionData> subs;
        subs.reserve(inputs.size());
        for (const auto& p : inputs) {
            subs.push_back(load_submission(p));
        }

        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("Erro ao abrir arquivo de saída: " + output_path);
        }
        out << "id,x,y,deg\n";

        double total_score = 0.0;

        for (int n = 1; n <= 200; ++n) {
            int best_k = -1;
            double best_side = std::numeric_limits<double>::infinity();
            std::vector<TreePose> best_poses;

            for (int k = 0; k < static_cast<int>(subs.size()); ++k) {
                const auto& recs = subs[static_cast<size_t>(k)].by_n[static_cast<size_t>(n)];
                std::vector<TreePose> poses;
                poses.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    const auto& r = recs[static_cast<size_t>(i)];
                    poses.push_back(TreePose{r.x, r.y, r.deg});
                }
                if (any_overlap(base_poly, poses, radius)) {
                    continue;
                }
                double side = side_for_solution(base_poly, poses);
                if (side + 1e-15 < best_side) {
                    best_side = side;
                    best_k = k;
                    best_poses = std::move(poses);
                }
            }

            if (best_k < 0) {
                throw std::runtime_error("Nenhum input válido para n=" + std::to_string(n));
            }

            std::vector<TreePose> out_poses = best_poses;
            if (final_rigid) {
                std::vector<TreePose> rigid_sol = out_poses;
                optimize_rigid_rotation(base_poly, rigid_sol);
                auto rigid_q = quantize_poses(rigid_sol);
                if (!any_overlap(base_poly, rigid_q, radius)) {
                    double rigid_side = side_for_solution(base_poly, rigid_q);
                    if (rigid_side + 1e-15 < best_side) {
                        best_side = rigid_side;
                        out_poses = std::move(rigid_q);
                    }
                }
            }

            total_score += (best_side * best_side) / static_cast<double>(n);

            for (int i = 0; i < n; ++i) {
                const auto& p = out_poses[static_cast<size_t>(i)];
                out << std::setw(3) << std::setfill('0') << n << "_" << i << ","
                    << fmt_submission_value(p.x) << ","
                    << fmt_submission_value(p.y) << ","
                    << fmt_submission_value(p.deg) << "\n";
            }
        }

        std::cout << "Submission ensembling gerada em " << output_path << "\n";
        std::cout << "Inputs: " << inputs.size() << "\n";
        std::cout << "Score (local): " << std::fixed << std::setprecision(9)
                  << total_score << "\n";
        std::cout << "Final rigid: " << (final_rigid ? "on" : "off") << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
