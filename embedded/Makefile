# SOFIA ENGINE — embedded reference runtime build
# Copyright 2026 Rootcastle Engineering & Innovation. Apache-2.0.
#
# Host build only. This Makefile does NOT cross-compile for any MCU target;
# see docs/embedded.md for target status (all targets are NOT VERIFIED here).

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow \
           -Wstrict-prototypes -fno-common -O2
LDLIBS  ?= -lm
SRC      = src/sofia_features.c
TEST     = tests/test_sofia_features.c
BIN      = build/test_sofia_features

.PHONY: all test clean analyze

all: $(BIN)

$(BIN): $(SRC) $(TEST) src/sofia_features.h src/sofia_fixed.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $(BIN) $(TEST) $(SRC) $(LDLIBS)

test: $(BIN)
	$(BIN)

# Strict analysis pass. Requires clang; skipped if unavailable.
analyze:
	@command -v clang >/dev/null 2>&1 && \
	  clang --analyze -std=c99 -I src $(SRC) $(TEST) || \
	  echo "clang not available; static analysis skipped"

clean:
	rm -rf build
