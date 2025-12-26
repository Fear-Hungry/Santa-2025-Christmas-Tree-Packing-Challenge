#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "santa2025/simulated_annealing.hpp"
#include "santa2025/submission_csv.hpp"
#include "santa2025/tree_polygon.hpp"

namespace {

struct Args {
    std::vector<std::string> inputs;
    std::string out_csv;
    std::string out_json;
    int nmax = 200;
    int csv_precision = 17;
    double eps = 1e-12;
};

std::vector<std::string> parse_csv_strings(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
            continue;
        }
        out.push_back(tok);
    }
    return out;
}

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

        if (a == "--in") {
            const auto more = parse_csv_strings(need("--in"));
            args.inputs.insert(args.inputs.end(), more.begin(), more.end());
        } else if (a == "--out") {
            args.out_csv = need("--out");
        } else if (a == "--out-json") {
            args.out_json = need("--out-json");
        } else if (a == "--nmax") {
            args.nmax = std::stoi(need("--nmax"));
        } else if (a == "--csv-precision") {
            args.csv_precision = std::stoi(need("--csv-precision"));
        } else if (a == "--eps") {
            args.eps = std::stod(need("--eps"));
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: merge_submissions --out merged.csv [--in a.csv,b.csv] [a.csv b.csv ...]\n"
                      << "                        [--nmax 200] [--out-json report.json] [--csv-precision 17] [--eps 1e-12]\n";
            std::exit(0);
        } else if (!a.empty() && a[0] == '-') {
            throw std::runtime_error("unknown arg: " + a);
        } else {
            args.inputs.push_back(a);
        }
    }

    if (args.inputs.size() < 2) {
        throw std::runtime_error("need at least 2 input CSVs (pass via --in or as positional args)");
    }
    if (args.out_csv.empty()) {
        throw std::runtime_error("missing --out <merged.csv>");
    }
    if (args.nmax <= 0 || args.nmax > 200) {
        throw std::runtime_error("--nmax must be in [1,200]");
    }
    if (args.csv_precision < 0 || args.csv_precision > 17) {
        throw std::runtime_error("--csv-precision must be in [0,17]");
    }
    if (!(args.eps > 0.0)) {
        throw std::runtime_error("--eps must be > 0");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);

        std::vector<santa2025::Submission> subs;
        subs.reserve(args.inputs.size());

        santa2025::ReadSubmissionOptions ropt;
        ropt.nmax = args.nmax;
        ropt.require_complete = true;

        for (const auto& path : args.inputs) {
            std::ifstream f(path);
            if (!f) {
                throw std::runtime_error("failed to open input: " + path);
            }
            subs.push_back(santa2025::read_submission_csv(f, ropt));
        }

        const santa2025::Polygon tree = santa2025::tree_polygon();

        santa2025::Submission merged;
        merged.poses.resize(static_cast<size_t>(args.nmax + 1));

        struct Pick {
            int puzzle = 0;
            int picked_input = 0;
            double s = 0.0;
            double term = 0.0;
        };
        std::vector<Pick> picks;
        picks.reserve(static_cast<size_t>(args.nmax));

        double total = 0.0;

        for (int puzzle = 1; puzzle <= args.nmax; ++puzzle) {
            int best_k = 0;
            double best_s = std::numeric_limits<double>::infinity();

            for (int k = 0; k < static_cast<int>(subs.size()); ++k) {
                const auto& poses = subs[static_cast<size_t>(k)].poses[static_cast<size_t>(puzzle)];
                const double s = santa2025::packing_s200(tree, poses);
                if (s < best_s) {
                    best_s = s;
                    best_k = k;
                }
            }

            const double term = (best_s * best_s) / static_cast<double>(puzzle);
            total += term;

            merged.poses[static_cast<size_t>(puzzle)] = subs[static_cast<size_t>(best_k)].poses[static_cast<size_t>(puzzle)];
            picks.push_back(Pick{puzzle, best_k, best_s, term});
        }

        {
            std::ofstream f(args.out_csv);
            if (!f) {
                throw std::runtime_error("failed to open --out file: " + args.out_csv);
            }
            santa2025::write_submission_csv(merged, f, args.nmax, args.csv_precision);
        }

        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\n";
        out << "  \"nmax\": " << args.nmax << ",\n";
        out << "  \"score\": " << total << ",\n";
        out << "  \"out_csv\": \"" << args.out_csv << "\",\n";
        out << "  \"inputs\": [\n";
        for (size_t i = 0; i < args.inputs.size(); ++i) {
            out << "    \"" << args.inputs[i] << "\"";
            if (i + 1 != args.inputs.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ],\n";
        out << "  \"picked\": [\n";
        for (size_t i = 0; i < picks.size(); ++i) {
            const auto& p = picks[i];
            out << "    {\"puzzle\": " << p.puzzle << ", \"input\": " << p.picked_input << ", \"s\": " << p.s
                << ", \"term\": " << p.term << "}";
            if (i + 1 != picks.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";

        const std::string payload = out.str();
        std::cout << payload;

        if (!args.out_json.empty()) {
            std::ofstream jf(args.out_json);
            if (!jf) {
                throw std::runtime_error("failed to open --out-json: " + args.out_json);
            }
            jf << payload;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
