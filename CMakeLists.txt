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
include_directories(${CMAKE_SOURCE_DIR}/include)

set(SRC_COMMON
  src/geometry/geom.cpp
  src/geometry/collision.cpp
  src/geometry/collision_polygons.cpp
  src/utils/micro_adjust.cpp
  src/utils/score.cpp
)
set(SRC_COMPACTION_CONTACT src/solvers/compaction_contact.cpp)
set(SRC_SUBMISSION src/utils/submission_io.cpp)
set(SRC_SA
  src/solvers/sa/sa.cpp
  src/solvers/sa/sa_refine.cpp
)
set(SRC_PREFIX_PRUNE src/solvers/prefix_prune.cpp)
set(SRC_BOUNDARY_REFINE src/solvers/boundary_refine.cpp)
set(SRC_TILING_POOL src/solvers/tiling_pool.cpp)
set(SRC_GA src/solvers/ga.cpp)
set(SRC_BASELINE src/solvers/baseline.cpp)
set(SRC_SOLVER_TILE_CLI src/utils/solver_tile_cli.cpp)
set(SRC_ENSEMBLE_CLI src/utils/ensemble_submissions_cli.cpp)

add_executable(solver_tile
  apps/solver_tile.cpp
  ${SRC_COMMON}
  ${SRC_COMPACTION_CONTACT}
  ${SRC_SUBMISSION}
  ${SRC_SA}
  ${SRC_PREFIX_PRUNE}
  ${SRC_BOUNDARY_REFINE}
  ${SRC_TILING_POOL}
  ${SRC_SOLVER_TILE_CLI}
)

add_executable(solver_tessellation
  apps/solver_tessellation.cpp
  ${SRC_COMMON}
  ${SRC_COMPACTION_CONTACT}
  ${SRC_SUBMISSION}
  ${SRC_SA}
  ${SRC_GA}
)

add_executable(blend_repair
  apps/blend_repair.cpp
  ${SRC_COMMON}
  ${SRC_SUBMISSION}
  ${SRC_SA}
)

add_executable(ensemble_submissions
  apps/ensemble_submissions.cpp
  ${SRC_COMMON}
  ${SRC_SUBMISSION}
  ${SRC_ENSEMBLE_CLI}
)

add_executable(score_submission
  apps/score_submission.cpp
  ${SRC_COMMON}
  ${SRC_SUBMISSION}
)

add_executable(solver_baseline
  apps/solver_baseline.cpp
  ${SRC_COMMON}
  ${SRC_SUBMISSION}
  ${SRC_BASELINE}
)

add_executable(compact_contact
  apps/compact_contact.cpp
  ${SRC_COMMON}
  ${SRC_COMPACTION_CONTACT}
  ${SRC_SUBMISSION}
)

add_executable(tile_density_search
  apps/tile_density_search.cpp
  ${SRC_COMMON}
  ${SRC_SUBMISSION}
  ${SRC_SA}
  ${SRC_PREFIX_PRUNE}
  ${SRC_BOUNDARY_REFINE}
  ${SRC_TILING_POOL}
)
