#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "santa2025/nfp.hpp"

namespace santa2025 {

struct ReadSubmissionOptions {
    int nmax = 200;
    bool require_complete = true;  // require all puzzles 1..nmax and indices 0..puzzle-1
};

struct Submission {
    // poses[p] has size p for p in [1..nmax]. poses[0] is unused.
    std::vector<std::vector<Pose>> poses;
};

std::string make_submission_id(int puzzle, int index);
std::string format_submission_value(double v, int precision = 17);
double parse_submission_value(std::string_view token);

void write_prefix_submission_csv(std::ostream& out, const std::vector<Pose>& poses, int nmax = 200, int precision = 17);
void write_submission_csv(const Submission& sub, std::ostream& out, int nmax = 200, int precision = 17);

Submission read_submission_csv(std::istream& in, const ReadSubmissionOptions& opt = {});

}  // namespace santa2025
