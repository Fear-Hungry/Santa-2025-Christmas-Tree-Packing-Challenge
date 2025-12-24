cmake_minimum_required(VERSION 3.16)
project(santa_2025_packing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/bin")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
if(CMAKE_CONFIGURATION_TYPES)
  foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${OUTPUT_DIR}")
  endforeach()
endif()

add_compile_options(-O2)

set(SRC_GEOMETRY
  src/geometry/geom.cpp
  src/geometry/collision.cpp
  src/geometry/collision_polygons.cpp
)

set(SRC_UTILS
  src/utils/micro_adjust.cpp
  src/utils/score.cpp
  src/utils/submission_io.cpp
  src/utils/ensemble_submissions_cli.cpp
)

set(SRC_SOLVERS
  src/solvers/baseline.cpp
  src/solvers/boundary_refine.cpp
  src/solvers/compaction_contact.cpp
  src/solvers/ga.cpp
  src/solvers/post_opt.cpp
  src/solvers/prefix_prune.cpp
  src/solvers/tiling_pool.cpp
  src/solvers/repair.cpp
  src/solvers/sa/sa.cpp
  src/solvers/sa/sa_refine.cpp
  src/utils/solver_tile_cli.cpp
)

add_library(santa_geometry ${SRC_GEOMETRY})
target_include_directories(santa_geometry PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(santa_utils ${SRC_UTILS})
target_include_directories(santa_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(santa_utils PUBLIC santa_geometry)

add_library(santa_solvers ${SRC_SOLVERS})
target_include_directories(santa_solvers PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(santa_solvers PUBLIC santa_geometry santa_utils)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(santa_solvers PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(solver_tile apps/solver_tile.cpp)
target_link_libraries(solver_tile PRIVATE santa_solvers)

add_executable(solver_tessellation apps/solver_tessellation.cpp)
target_link_libraries(solver_tessellation PRIVATE santa_solvers)

add_executable(blend_repair apps/blend_repair.cpp)
target_link_libraries(blend_repair PRIVATE santa_solvers)

add_executable(ensemble_submissions apps/ensemble_submissions.cpp)
target_link_libraries(ensemble_submissions PRIVATE santa_utils)

add_executable(score_submission apps/score_submission.cpp)
target_link_libraries(score_submission PRIVATE santa_utils)

add_executable(solver_baseline apps/solver_baseline.cpp)
target_link_libraries(solver_baseline PRIVATE santa_solvers)

add_executable(compact_contact apps/compact_contact.cpp)
target_link_libraries(compact_contact PRIVATE santa_solvers)

add_executable(tile_density_search apps/tile_density_search.cpp)
target_link_libraries(tile_density_search PRIVATE santa_solvers)

add_executable(post_opt apps/post_opt.cpp)
target_link_libraries(post_opt PRIVATE santa_solvers)
