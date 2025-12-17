#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"

struct InstanceData {
    std::vector<TreePose> poses;
};

struct BreakdownRow {
    int n = 0;
    double s_n = 0.0;
    double term = 0.0;  // s_n^2 / n
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

int main(int argc, char** argv) {
    try {
        std::string path = "submission.csv";
        bool check_overlap = true;
        bool breakdown = false;
        bool breakdown_all = false;
        int top_k = 10;
        std::string csv_out;

        auto need = [&](int& i, const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Faltou valor para " + name);
            }
            return argv[++i];
        };
        auto parse_int = [&](const std::string& s, const std::string& name) -> int {
            size_t pos = 0;
            int v = std::stoi(s, &pos);
            if (pos != s.size()) {
                throw std::runtime_error("Inteiro inválido para " + name + ": " + s);
            }
            return v;
        };

        bool path_set = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) != 0) {
                if (path_set) {
                    throw std::runtime_error("Argumento extra inesperado: " + arg);
                }
                path = arg;
                path_set = true;
                continue;
            }

            if (arg == "--no-overlap") {
                check_overlap = false;
            } else if (arg == "--breakdown") {
                breakdown = true;
            } else if (arg == "--all") {
                breakdown = true;
                breakdown_all = true;
            } else if (arg == "--top") {
                breakdown = true;
                top_k = parse_int(need(i, arg), arg);
            } else if (arg == "--csv") {
                breakdown = true;
                csv_out = need(i, arg);
            } else {
                throw std::runtime_error("Argumento desconhecido: " + arg);
            }
        }

        if (top_k < 0) {
            throw std::runtime_error("--top precisa ser >= 0.");
        }

        std::ifstream in(path);
        if (!in) {
            std::cerr << "Erro ao abrir arquivo de submissão: " << path
                      << "\n";
            return 1;
        }

        std::string header;
        if (!std::getline(in, header)) {
            std::cerr << "Arquivo vazio: " << path << "\n";
            return 1;
        }

        std::map<int, InstanceData> instances;

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            std::string id, sx, sy, sdeg;
            if (!parse_submission_line(line, id, sx, sy, sdeg)) {
                throw std::runtime_error("Linha inválida: " + line);
            }

            if (id.size() < 5 || id[3] != '_') {
                throw std::runtime_error("Formato de id inválido: " + id);
            }

            int n = std::stoi(id.substr(0, 3));
            int idx = std::stoi(id.substr(4));

            double x = parse_prefixed_value(sx);
            double y = parse_prefixed_value(sy);
            double deg = parse_prefixed_value(sdeg);

            if (x < -100.0 || x > 100.0 || y < -100.0 || y > 100.0) {
                throw std::runtime_error("Coordenada fora de [-100,100] em id: " +
                                         id);
            }

            auto& inst = instances[n];
            if (static_cast<int>(inst.poses.size()) <= idx) {
                inst.poses.resize(idx + 1);
            }
            inst.poses[idx] = TreePose{x, y, deg};
        }

        Polygon base_poly = get_tree_polygon();

        const int max_n = 200;
        double total_score = 0.0;
        std::vector<BreakdownRow> rows;
        rows.reserve(static_cast<size_t>(max_n));

        for (int n = 1; n <= max_n; ++n) {
            auto it = instances.find(n);
            if (it == instances.end()) {
                throw std::runtime_error("Instância n=" + std::to_string(n) +
                                         " ausente no submission.");
            }

            const auto& poses = it->second.poses;
            if (static_cast<int>(poses.size()) != n) {
                throw std::runtime_error(
                    "Instância n=" + std::to_string(n) +
                    " não possui exatamente " + std::to_string(n) +
                    " árvores (size=" + std::to_string(poses.size()) + ").");
            }

            // Checagem opcional de colisão local (similar à validação do
            // Kaggle).
            if (check_overlap && any_overlap(base_poly, poses)) {
                throw std::runtime_error(
                    "Overlap detectado na instância n=" + std::to_string(n) +
                    ".");
            }

            auto polys = transformed_polygons(base_poly, poses);
            double sn = bounding_square_side(polys);
            double term = (sn * sn) / static_cast<double>(n);
            total_score += term;
            rows.push_back(BreakdownRow{n, sn, term});
        }

        std::cout << "Score: " << std::fixed << std::setprecision(9)
                  << total_score << "\n";

        if (breakdown) {
            std::sort(rows.begin(),
                      rows.end(),
                      [](const BreakdownRow& a, const BreakdownRow& b) {
                          if (a.term != b.term) {
                              return a.term > b.term;
                          }
                          if (a.s_n != b.s_n) {
                              return a.s_n > b.s_n;
                          }
                          return a.n < b.n;
                      });

            const int k = breakdown_all ? static_cast<int>(rows.size())
                                        : std::min(top_k, static_cast<int>(rows.size()));
            std::cout << "Top " << k << " (por s_n^2/n):\n";
            std::cout << "rank,n,s_n,term\n";
            for (int i = 0; i < k; ++i) {
                const auto& r = rows[static_cast<size_t>(i)];
                std::cout << (i + 1) << ","
                          << r.n << ","
                          << std::fixed << std::setprecision(9) << r.s_n << ","
                          << std::fixed << std::setprecision(12) << r.term << "\n";
            }

            if (!csv_out.empty()) {
                std::ofstream out(csv_out);
                if (!out) {
                    throw std::runtime_error("Erro ao abrir --csv: " + csv_out);
                }
                out << "n,s_n,term\n";
                for (const auto& r : rows) {
                    out << r.n << ","
                        << std::fixed << std::setprecision(9) << r.s_n << ","
                        << std::fixed << std::setprecision(12) << r.term << "\n";
                }
                std::cout << "Breakdown salvo em " << csv_out << "\n";
            }
        }

    } catch (const std::exception& ex) {
        std::cerr << "Erro no simulador de score: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
