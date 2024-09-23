#!/bin/bash
ncu -o test -f --replay-mode kernel --kernel-name "regex:^[^n][^c].*" --launch-skip 50 --launch-count 30 --device 0 python matmul-overlap.py