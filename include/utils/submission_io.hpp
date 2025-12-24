#pragma once

#include <string>
#include <vector>

#include "geometry/geom.hpp"

struct SubmissionPoses {
    std::string path;
    std::vector<std::vector<TreePose>> by_n;  // index 0..n_max
};

bool parse_submission_line(const std::string& line,
                           std::string& id,
                           std::string& sx,
                           std::string& sy,
                           std::string& sdeg);

double parse_prefixed_value(const std::string& s);

std::string fmt_submission_id(int n, int idx);

SubmissionPoses load_submission_poses(const std::string& path, int n_max = 200);

double quantize_value(double x, int decimals = 9);

TreePose quantize_pose(const TreePose& pose, int decimals = 9);

TreePose quantize_pose_wrap_deg(const TreePose& pose, int decimals = 9);

std::vector<TreePose> quantize_poses(const std::vector<TreePose>& poses,
                                     int decimals = 9);

std::vector<TreePose> quantize_poses_wrap_deg(const std::vector<TreePose>& poses,
                                              int decimals = 9);

