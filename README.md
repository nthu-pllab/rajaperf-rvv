# RAJAPerf with RVV

RAJAPerf v2025.03.0 extended with a `RAJA_RVV` variant that runs on RISC-V vector
hardware through the RVV-enabled RAJA fork at
https://github.com/ShouChenChiu/RAJA (branch `develop`).

## What is here

| File | Purpose |
|---|---|
| `rajaperf-v2025.03.0-rvv-variants.patch` | Adds the `RAJA_RVV` variant to RAJAPerf v2025.03.0 |
| `build.sh` | Downloads RAJAPerf, applies the patch, pulls in the RVV RAJA fork, cross-builds a static `raja-perf.exe` |
| `run.sh` | Runs the five RVV kernels on the board, one result directory per kernel |
| `results-2025-10-k1/` | Original measurements on a SpacemiT K1 (BPI-F3), Oct 2025 |
| `results-2026-10-05-recheck/` | Re-verification run from a fresh build, Oct 2026 |

## Kernels with an RVV variant

`Basic_DAXPY`, `Polybench_GEMM`, `Polybench_GESUMMV`, `Polybench_2MM`, `Polybench_3MM`.
All other RAJAPerf kernels only have the stock variants.

## How the RVV support works

The fork adds an RVV backend to RAJA's tensor-register layer
(`RAJA::expt::VectorRegister`, `SquareMatrixRegister`). It is **not** a vectorizing
`forall` policy. The `RAJA_RVV` kernel variants are written against that layer.

The backend is switched on at compile time:

| Flag | Meaning |
|---|---|
| `-D__RVVF__` | enable the RVV backend |
| `-D__VL__=256` | VLEN in bits of the target (must match the hardware) |
| `-D__LMUL__=1` | register grouping, 1 / 2 / 4 / 8 |
| `-march=rv64gcv_zvl256b -mrvv-vector-bits=zvl` | compiler must agree on VLEN |

C++17 is required. Tested with GCC 15.1 (riscv64-unknown-linux-gnu).

## Build and run

```bash
# host: build.sh [LMUL] [VLEN_BITS] [NOVEC]
RVGCC_PREFIX=/path/to/riscv64-unknown-linux-gnu- ./build.sh 1 256
# -> work/RAJAPerf-v2025.03.0/build-lmul1/bin/raja-perf.exe (static)

# board
./run.sh ./raja-perf.exe results
```

`raja-perf.exe` takes the usual RAJAPerf options, e.g.
`-k Polybench_GEMM -v Base_Seq RAJA_Seq RAJA_RVV`.
Note: Polybench_GEMM defaults to 4 reps and 2MM/3MM to 2, so `--repfact` below 0.5
gives 0 reps and a 0 s result.

## Results on SpacemiT K1 (VLEN 256), default sizes, mean seconds

| Kernel | Scalar (Base_Seq) | RVV LMUL1 | LMUL2 | LMUL4 | LMUL8 |
|---|---|---|---|---|---|
| Polybench_GEMM | 331.1 | 59.0 | 30.7 | 21.7 | 24.2 |
| Polybench_2MM | 287.2 | 52.1 | 26.6 | 18.2 | - |
| Polybench_3MM | 440.0 | 78.0 | 43.0 | 28.0 | 30.5 |
| Basic_DAXPY | 2.74 | 2.26 | 2.23 | 2.24 | 2.26 |
| Polybench_GESUMMV | 0.60 | 1.95 | - | 1.32 | - |

Matrix kernels: 5.6x (LMUL1) to 15x (LMUL4) over scalar. DAXPY is memory-bound
(1.2x). GESUMMV's RVV variant is slower than scalar. All RVV checksums match
Base_Seq. A fresh build in Oct 2026 reproduced the GEMM result (5.5x).

## Caveats for other targets (K3, gem5)

- `__VL__` is fixed at compile time; set it to the target's VLEN.
- Reductions in the register code assume 64-bit elements at VLEN 256.
- Only the five kernels above are RVV-aware.
