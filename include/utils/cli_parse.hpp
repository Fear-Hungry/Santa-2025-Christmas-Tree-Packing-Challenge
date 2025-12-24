#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::string require_arg(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag + ".");
    }
    return argv[++i];
}

inline int parse_int(const std::string& s) {
    size_t pos = 0;
    int v = std::stoi(s, &pos);
    if (pos != s.size()) {
        throw std::runtime_error("Invalid integer: " + s);
    }
    return v;
}

inline uint64_t parse_u64(const std::string& s) {
    size_t pos = 0;
    uint64_t v = std::stoull(s, &pos);
    if (pos != s.size()) {
        throw std::runtime_error("Invalid uint64: " + s);
    }
    return v;
}

inline double parse_double(const std::string& s) {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) {
        throw std::runtime_error("Invalid double: " + s);
    }
    return v;
}

inline std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        out.push_back(parse_double(item));
    }
    return out;
}
