CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Iinclude
LDFLAGS ?=

BIN_DIR := bin

SRC_COMMON := src/geom.cpp src/collision.cpp src/collision_polygons.cpp \
	src/micro_adjust.cpp
SRC_SUBMISSION := src/submission_io.cpp
SRC_SA := src/sa.cpp
SRC_PREFIX_PRUNE := src/prefix_prune.cpp
SRC_BOUNDARY_REFINE := src/boundary_refine.cpp
SRC_TILING_POOL := src/tiling_pool.cpp
SRC_GA := src/ga.cpp
SRC_BASELINE := src/baseline.cpp

SOLVER_TILE_SRC := $(SRC_COMMON) $(SRC_SUBMISSION) $(SRC_SA) \
	$(SRC_PREFIX_PRUNE) $(SRC_BOUNDARY_REFINE) $(SRC_TILING_POOL) \
	src/solver_tile_cli.cpp
SOLVER_TESSELLATION_SRC := $(SRC_COMMON) $(SRC_SUBMISSION) $(SRC_SA) $(SRC_GA)
BLEND_REPAIR_SRC := $(SRC_COMMON) $(SRC_SUBMISSION) $(SRC_SA)
ENSEMBLE_SRC := $(SRC_COMMON) $(SRC_SUBMISSION) src/ensemble_submissions_cli.cpp
SCORE_SRC := $(SRC_COMMON) $(SRC_SUBMISSION)
BASELINE_SRC := $(SRC_COMMON) $(SRC_SUBMISSION) $(SRC_BASELINE)
COMPACT_CONTACT_SRC := $(SRC_COMMON) $(SRC_SUBMISSION)
TILE_DENSITY_SRC := $(SRC_COMMON) $(SRC_SUBMISSION) $(SRC_SA) \
	$(SRC_PREFIX_PRUNE) $(SRC_BOUNDARY_REFINE) $(SRC_TILING_POOL)

.PHONY: all clean regress regress-record \
	solver_tile solver_tessellation blend_repair ensemble_submissions \
	score_submission solver_baseline compact_contact tile_density_search

all: solver_tile solver_tessellation blend_repair ensemble_submissions \
	score_submission solver_baseline compact_contact tile_density_search

solver_tile: $(BIN_DIR)/solver_tile
solver_tessellation: $(BIN_DIR)/solver_tessellation
blend_repair: $(BIN_DIR)/blend_repair
ensemble_submissions: $(BIN_DIR)/ensemble_submissions
score_submission: $(BIN_DIR)/score_submission
solver_baseline: $(BIN_DIR)/solver_baseline
compact_contact: $(BIN_DIR)/compact_contact
tile_density_search: $(BIN_DIR)/tile_density_search

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/solver_tile: apps/solver_tile.cpp $(SOLVER_TILE_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/solver_tessellation: apps/solver_tessellation.cpp $(SOLVER_TESSELLATION_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/blend_repair: apps/blend_repair.cpp $(BLEND_REPAIR_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/ensemble_submissions: apps/ensemble_submissions.cpp $(ENSEMBLE_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/score_submission: apps/score_submission.cpp $(SCORE_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/solver_baseline: apps/solver_baseline.cpp $(BASELINE_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/compact_contact: apps/compact_contact.cpp $(COMPACT_CONTACT_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/tile_density_search: apps/tile_density_search.cpp $(TILE_DENSITY_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

regress: $(BIN_DIR)/solver_tile $(BIN_DIR)/solver_baseline $(BIN_DIR)/score_submission
	./scripts/check_regression.sh

regress-record: $(BIN_DIR)/solver_tile $(BIN_DIR)/solver_baseline $(BIN_DIR)/score_submission
	./scripts/check_regression.sh --record

clean:
	rm -f $(BIN_DIR)/solver_tile \
		$(BIN_DIR)/solver_tessellation \
		$(BIN_DIR)/blend_repair \
		$(BIN_DIR)/ensemble_submissions \
		$(BIN_DIR)/score_submission \
		$(BIN_DIR)/solver_baseline \
		$(BIN_DIR)/compact_contact \
		$(BIN_DIR)/tile_density_search
