# NOTE: CMake is the source of truth for this repository.
# This Makefile is a small convenience wrapper used by tooling (e.g. santa-lab /
# santa-pipeline) that expects `make` to exist at the repo root.
#
# The actual wrapper rules live in `convenience.mk`.

include convenience.mk
