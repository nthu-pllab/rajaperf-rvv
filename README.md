# RAJA RVV + RAJAPerf: what exists and how to reproduce it

Re-verified 2026-10-05 on the BPI-F3 (SpacemiT K1, X60 cores, RVV 1.0, VLEN=256).

## Repositories

| What | Where | Visibility |
|---|---|---|
| RAJA with RVV tensor-register backend | https://github.com/ShouChenChiu/RAJA (branch `develop`, head a8cdcd7, Jun 2025) | public |
| Same RVV backend, older guard (`__riscv_vector`) | `git@github.com:nthu-pllab/RAJA.git` (`main`, ed431a8) | private (NTHU org) |
| RAJAPerf v2025.03.0 with `RAJA_RVV` variants | `git@github.com:nthu-pllab/RAJAPerf-v2025.03.0.git` (`main`, 13c7d4b "Add RAJA_RVV variants for selected kernels") | private (NTHU org) |
| Local working copies | `~/riscv-summit25/{RAJA,raja_rvv,RAJAPerf-v2025.03.0}` | this machine |

The public fork and the private nthu-pllab copy contain byte-identical register
implementations (`rvv_double/float/int32/int64.hpp`); they differ only in how
the backend is switched on (public fork: `-D__RVVF__`; nthu-pllab: automatic
under `__riscv_vector`) and in cosmetic edits to `traits.hpp`.

Since the RAJAPerf changes live in a private repo, this directory carries them
as a plain patch (`rajaperf-v2025.03.0-rvv-variants.patch`, 374 lines) that
applies to the upstream v2025.03.0 release tarball.

## What the RVV support is

It is **not** a vectorizing `forall` policy. It is a backend for RAJA's
experimental tensor-register layer (`RAJA::expt::VectorRegister`,
`SquareMatrixRegister`, `RAJA_TENSOR_REGISTER_TYPE`), implemented with RVV
intrinsics over fixed-length vector types
(`__attribute__((riscv_rvv_vector_bits(VL*LMUL)))`). Files:

```
include/RAJA/policy/tensor/arch.hpp            rvv_register policy, default register type
include/RAJA/policy/tensor/arch_impl.hpp       #include "arch/rvvf.hpp" under __RVVF__
include/RAJA/policy/tensor/arch/rvvf.hpp
include/RAJA/policy/tensor/arch/rvv/traits.hpp RegisterTraits: s_num_elem = VL*LMUL/elem_bits
include/RAJA/policy/tensor/arch/rvv/rvv_{double,float,int32,int64}.hpp
```

Compile-time knobs: `-D__RVVF__` (enable), `-D__VL__=256` (VLEN bits),
`-D__LMUL__=1|2|4|8`. The compiler must agree: `-march=rv64gcv_zvl256b
-mrvv-vector-bits=zvl`, and C++17 is required (`if constexpr`).

RAJAPerf kernels with a `RAJA_RVV` variant (written against that layer):
`Basic_DAXPY`, `Polybench_GEMM`, `Polybench_GESUMMV`, `Polybench_2MM`,
`Polybench_3MM`. All other kernels only have the stock variants.

## Scripts

- `build.sh [LMUL] [VLEN] [NOVEC]` - downloads the v2025.03.0 tarball, applies the
  patch, swaps in the public RAJA fork (with its own camp/blt/desul), configures
  and cross-builds a static `raja-perf.exe`. Tested end to end on 2026-10-05
  with GCC 15.1 (`~/rvv/riscv`).
- `run.sh <exe> <outdir> [REPFACT] [SIZEFACT]` - board-side driver producing one
  result directory per kernel (same layout as the Oct 2025 data).

Build flags used for the Oct 2025 numbers (from the CMake caches):

```
riscv64-unknown-linux-gnu-g++ (GCC 15.1.0)
-std=c++17 -O3 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl -mabi=lp64d -D__LMUL__={1,2,4,8}
novec baseline build adds: -fno-tree-vectorize
```

## Oct 2025 results, BPI-F3 / K1, default RAJAPerf sizes (mean seconds)

Raw CSVs under `results-2025-10-k1/` (copied from the board's `~/experiment/`).

| Kernel | Base_Seq | RAJA_Seq | RAJA_RVV LMUL1 | LMUL2 | LMUL4 | LMUL8 |
|---|---|---|---|---|---|---|
| Basic_DAXPY | 2.74 | 2.73 | 2.26 | 2.23 | 2.24 | 2.26 |
| Polybench_GESUMMV | 0.60 | 0.62 | 1.95 | - | 1.32 | - |
| Polybench_GEMM | 331.1 | 331.4 | 59.0 | 30.7 | 21.7 | 24.2 |
| Polybench_2MM | 287.2 | 288.9 | 52.1 | 26.6 | 18.2 | - |
| Polybench_3MM | 440.0 | 437.3 | 78.0 | 43.0 | 28.0 | 30.5 |

Takeaways: matrix kernels through `SquareMatrixRegister` get 5.6x (LMUL1) to
15x (LMUL4) over scalar; DAXPY is memory-bound (1.2x); GESUMMV's RVV variant is
slower than scalar (gather/reduction path). Checksums of all RVV variants match
Base_Seq to ~1e-13.

## 2026-10-05 re-verification

- Old LMUL1 binary (`~/experiment/lmul1/raja-perf.exe` on the board) still runs;
  DAXPY/GESUMMV checksums match.
- Fresh build from the **public fork** + patch (`build.sh 1 256 0`) builds and
  runs on the board; checksums match for all five kernels (binary kept as
  `raja-perf-pubfork-lmul1-vl256.exe`). Raw output in `results-2026-10-05-recheck/`.

| Kernel (LMUL1, public fork) | size/rep factor | Base_Seq s | RAJA_RVV s | speedup |
|---|---|---|---|---|
| Polybench_GEMM | 1.0 / 0.25 (1 rep) | 83.4 | 15.1 | 5.5x (Oct 2025: 5.6x) |
| Polybench_2MM | 0.25 / 0.5 (1 rep) | 12.4 | 3.19 | 3.9x |
| Polybench_3MM | 0.25 / 0.5 (1 rep) | 19.3 | 4.76 | 4.0x |
| Basic_DAXPY | 0.25 / 0.2 | 0.120 | 0.117 | 1.0x |
| Polybench_GESUMMV | 0.25 / 0.2 | 0.030 | 0.066 | 0.45x |

Board was not idle during these runs (bluetoothd pinned at ~50% of one core), so
treat them as functional confirmation, not a clean re-measurement.

## Caveats for a K3 / gem5 study

- Fixed-VLEN design: `__VL__` must equal the target's VLEN (K1 = 256). Check K3's
  VLEN before reusing binaries; gem5 RVV VLEN is configurable.
- The reductions (`sum`, `max`, `min`) default `N` to `__LMUL__ * 4`, i.e. they
  assume 64-bit elements at VLEN=256.
- `createMask` only handles LMUL 1/2/4/8 via `if constexpr`.
- Only the five kernels above have an RVV variant; everything else in RAJAPerf
  is compiler auto-vectorization only.
