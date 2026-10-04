#!/usr/bin/env bash
# Run the RAJA_RVV study kernels on the RISC-V board, one output directory per kernel.
# This mirrors the layout used for the Oct 2025 BPI-F3 (SpacemiT K1) numbers:
#   <OUT>/<kernel>/RAJAPerf-{timing-Average,speedup-Average,checksum}.{csv,txt}
#
# Usage:  ./run.sh <path/to/raja-perf.exe> <OUT_DIR> [REPFACT] [SIZEFACT]
#   REPFACT  multiplier on default reps (Polybench kernels default to 4 reps,
#            2MM/3MM to 2; keep REPFACT >= 0.5 or they run 0 reps and report 0 s)
#   SIZEFACT fraction of default problem size (default 1.0)
#
# The binary is static; run it under `script` so stdout stays line-buffered when
# redirected. Interleave kernels and let the board cool between long runs if you
# care about thermal throttling (passively cooled K1 drifts 10-20%).
set -euo pipefail
EXE=${1:?raja-perf.exe path}
OUT=${2:?output dir}
REPFACT=${3:-1.0}
SIZEFACT=${4:-1.0}
EXE=$(readlink -f "$EXE")

KERNELS="Basic_DAXPY Polybench_GESUMMV Polybench_GEMM Polybench_2MM Polybench_3MM"
VARIANTS="Base_Seq Lambda_Seq RAJA_Seq RAJA_RVV"

for k in $KERNELS; do
  d=$OUT/$k
  mkdir -p "$d"
  echo "== $k -> $d"
  script -qec "$EXE -k $k -v $VARIANTS --repfact $REPFACT --sizefact $SIZEFACT --outdir $d" /dev/null \
    | grep -i "running\|DONE\|error\|bad input" || true
  cat "$d/RAJAPerf-timing-Average.csv"
done

echo
echo "== summary (mean seconds)"
for k in $KERNELS; do tail -n1 "$OUT/$k/RAJAPerf-timing-Average.csv"; done
echo "== checksum diffs vs Base_Seq"
grep -h "RAJA_RVV" $OUT/*/RAJAPerf-checksum.txt
