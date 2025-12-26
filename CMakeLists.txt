cmake_minimum_required(VERSION 3.20)

project(santa2025_tree_packing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(santa2025
    src/geometry.cpp
    src/spatial_hash_grid.cpp
    src/collision_index.cpp
    src/nfp.cpp
    src/bottom_left.cpp
    src/packing_stats.cpp
    src/grid_shake.cpp
    src/simulated_annealing.cpp
    src/lns.cpp
    src/hyper_heuristic.cpp
    src/order_hyper_heuristic.cpp
    src/submission_csv.cpp
)
target_include_directories(santa2025 PUBLIC include)

if (MSVC)
    target_compile_options(santa2025 PRIVATE /W4 /permissive-)
else()
    target_compile_options(santa2025 PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(tree_info apps/tree_info.cpp)
target_link_libraries(tree_info PRIVATE santa2025)

add_executable(nfp_sanity apps/nfp_sanity.cpp)
target_link_libraries(nfp_sanity PRIVATE santa2025)

add_executable(bottom_left_pack apps/bottom_left_pack.cpp)
target_link_libraries(bottom_left_pack PRIVATE santa2025)

add_executable(grid_shake_pack apps/grid_shake_pack.cpp)
target_link_libraries(grid_shake_pack PRIVATE santa2025)

add_executable(score_prefix apps/score_prefix.cpp)
target_link_libraries(score_prefix PRIVATE santa2025)

add_executable(score_submission apps/score_submission.cpp)
target_link_libraries(score_submission PRIVATE santa2025)

add_executable(sa_opt apps/sa_opt.cpp)
target_link_libraries(sa_opt PRIVATE santa2025)

set_target_properties(tree_info nfp_sanity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

add_executable(hh_opt apps/hh_opt.cpp)
target_link_libraries(hh_opt PRIVATE santa2025)

add_executable(order_opt apps/order_opt.cpp)
target_link_libraries(order_opt PRIVATE santa2025)

add_executable(solve_all apps/solve_all.cpp)
target_link_libraries(solve_all PRIVATE santa2025)

add_executable(merge_submissions apps/merge_submissions.cpp)
target_link_libraries(merge_submissions PRIVATE santa2025)

set_target_properties(bottom_left_pack grid_shake_pack score_prefix score_submission sa_opt hh_opt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

set_target_properties(order_opt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

set_target_properties(solve_all PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

set_target_properties(merge_submissions PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
