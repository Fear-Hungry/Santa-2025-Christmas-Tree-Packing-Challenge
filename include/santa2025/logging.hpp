#pragma once

#include <mutex>

namespace santa2025 {

// Global mutex to keep stderr logs from multiple threads readable (one line at a time).
inline std::mutex& log_mutex() {
    static std::mutex mu;
    return mu;
}

}  // namespace santa2025

