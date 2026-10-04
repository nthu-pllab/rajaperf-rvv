#!/usr/bin/env bash
# Build RAJAPerf v2025.03.0 with the RAJA_RVV variants on top of the public
# RVV-enabled RAJA fork (https://github.com/ShouChenChiu/RAJA, branch develop).
#
# Usage:
#   ./build.sh [LMUL] [VLEN_BITS] [NOVEC]
#     LMUL       1|2|4|8   register grouping for the RAJA_RVV variant (default 1)
#     VLEN_BITS  256       vector length the fixed-size types are compiled for (default 256)
#     NOVEC      0|1       1 = add -fno-tree-vectorize so Base_Seq/RAJA_Seq stay scalar (default 0)
#
# Environment:
#   RVGCC_PREFIX  cross-compiler prefix (default: $HOME/rvv/riscv/bin/riscv64-unknown-linux-gnu-)
#   WORK          working directory (default: ./work)
#   JOBS          make -j (default: nproc)
#
# Produces: $WORK/RAJAPerf-v2025.03.0/build-lmul${LMUL}[-novec]/bin/raja-perf.exe (static)
set -euo pipefail

LMUL=${1:-1}
VLEN=${2:-256}
NOVEC=${3:-0}
RVGCC_PREFIX=${RVGCC_PREFIX:-$HOME/rvv/riscv/bin/riscv64-unknown-linux-gnu-}
WORK=${WORK:-$(pwd)/work}
JOBS=${JOBS:-$(nproc)}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=$HERE/rajaperf-v2025.03.0-rvv-variants.patch

TARBALL_URL=https://github.com/LLNL/RAJAPerf/releases/download/v2025.03.0/RAJAPerf-v2025.03.0.tar.gz
RAJA_FORK=https://github.com/ShouChenChiu/RAJA.git

mkdir -p "$WORK"
cd "$WORK"

# 1. RAJAPerf v2025.03.0 release tarball (bundles blt, tpl/RAJA, etc.)
if [ ! -d RAJAPerf-v2025.03.0 ]; then
  [ -f RAJAPerf-v2025.03.0.tar.gz ] || curl -sSL -o RAJAPerf-v2025.03.0.tar.gz "$TARBALL_URL"
  tar xzf RAJAPerf-v2025.03.0.tar.gz
  # 2. Add the RAJA_RVV variant (DAXPY, Polybench GEMM/GESUMMV/2MM/3MM)
  (cd RAJAPerf-v2025.03.0 && patch -p1 < "$PATCH")
  # 3. Replace bundled RAJA with the RVV fork (its own camp/blt/desul are newer
  #    than the ones in the v2025.03.0 tarball, so take them from the fork too).
  rm -rf RAJAPerf-v2025.03.0/tpl/RAJA
  git clone --branch develop "$RAJA_FORK" RAJAPerf-v2025.03.0/tpl/RAJA
  (cd RAJAPerf-v2025.03.0/tpl/RAJA && git submodule update --init --depth 1 blt tpl/camp tpl/desul)
fi

cd RAJAPerf-v2025.03.0
BUILD=build-lmul${LMUL}
EXTRA=""
if [ "$NOVEC" = "1" ]; then BUILD=${BUILD}-novec; EXTRA="-fno-tree-vectorize"; fi

ARCH="-O3 -march=rv64gcv_zvl${VLEN}b -mrvv-vector-bits=zvl -mabi=lp64d $EXTRA"
# __RVVF__ turns on the RVV tensor-register backend in the fork; __VL__/__LMUL__
# size the fixed-length vector types (RegisterTraits::s_num_elem = VL*LMUL/elem_bits).
RVV_DEFS="-D__RVVF__ -D__VL__=${VLEN} -D__LMUL__=${LMUL}"

mkdir -p "$BUILD"
cd "$BUILD"
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER="${RVGCC_PREFIX}gcc" \
  -DCMAKE_CXX_COMPILER="${RVGCC_PREFIX}g++" \
  -DBLT_CXX_STD=c++17 \
  -DCMAKE_C_FLAGS="$ARCH" \
  -DCMAKE_CXX_FLAGS="$ARCH $RVV_DEFS" \
  -DCMAKE_EXE_LINKER_FLAGS="-static" \
  -DENABLE_OPENMP=OFF -DENABLE_TESTS=OFF -DRAJA_PERFSUITE_ENABLE_TESTS=OFF \
  .. > cmake.log
make -j"$JOBS" > make.log 2>&1
ls -la bin/raja-perf.exe
echo "OK: $(pwd)/bin/raja-perf.exe"
