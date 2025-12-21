# Minimal Makefile shim for santa-lab tooling (CMake-backed build).

BUILD_DIR ?= build
CMAKE ?= cmake
JOBS ?= 8

TARGETS := solver_tile \
	solver_tessellation \
	blend_repair \
	ensemble_submissions \
	score_submission \
	solver_baseline \
	compact_contact \
	tile_density_search

.PHONY: all $(TARGETS) clean

all: $(TARGETS)

$(BUILD_DIR)/CMakeCache.txt:
	$(CMAKE) -S . -B $(BUILD_DIR)

$(TARGETS): $(BUILD_DIR)/CMakeCache.txt
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS) --target $@

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean
