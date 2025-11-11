#!/bin/bash
# Helper script to run online EAGLE tests with correct LD_PRELOAD

LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so" \
python3 test_online_eagle_integration.py "$@"
