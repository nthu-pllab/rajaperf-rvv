// Tail-handling probe for the RVV tensor-register backend.
// Exercises partial (N < lanes) loads through the same paths RAJAPerf's
// RAJA_RVV kernels use: VectorRegister (DAXPY-style) and SquareMatrixRegister
// multiply (GEMM-style), with sizes that are NOT multiples of the lane count.
#include "RAJA/RAJA.hpp"
#include <riscv_vector.h>
#include <cmath>
#include <cstdio>
#include <cstring>

using vec_t = RAJA::expt::VectorRegister<double, RAJA::expt::rvv_register>;
using mat_t = RAJA::expt::SquareMatrixRegister<double, RAJA::expt::RowMajorLayout>;

static int bad = 0;
static void check(const char* what, double got, double want) {
  bool ok = std::fabs(got - want) < 1e-9 * (1 + std::fabs(want));
  std::printf("%-34s got=%-22.15g want=%-22.15g %s\n", what, got, want, ok ? "ok" : "MISMATCH");
  if (!ok) bad++;
}

int main() {
  constexpr int lanes = vec_t::s_num_elem;
  std::printf("lanes per register = %d\n", lanes);

  // 0. Raw intrinsic semantics: what do inactive lanes of a masked load hold?
  {
    size_t vl = __riscv_vsetvlmax_e64m1();
    double src[16]; for (int i = 0; i < 16; ++i) src[i] = 1.0;
    vuint64m1_t idx = __riscv_vid_v_u64m1(vl);
    vbool64_t m = __riscv_vmsltu_vx_u64m1_b64(idx, 1, vl);  // only lane 0 active
    vfloat64m1_t v = __riscv_vle64_v_f64m1_m(m, src, vl);    // agnostic policy (what the fork uses)
    double out[16]; __riscv_vse64_v_f64m1(out, v, vl);
    std::printf("masked-load (agnostic) inactive lanes:");
    for (size_t i = 1; i < vl; ++i) std::printf(" %g", out[i]);
    std::printf("\n");
  }

  // 1. Vector dot product with N = lanes+1 ... exercises load_packed_n + sum()
  for (int N = 1; N <= lanes; ++N) {
    double x[64], y[64], want = 0;
    for (int i = 0; i < N; ++i) { x[i] = i + 1; y[i] = 0.5 * i; want += x[i] * y[i]; }
    // Registers that previously held non-zero data (as they would after an
    // earlier tile in a real kernel): here, NaN, to make leakage obvious.
    vec_t a(std::nan("")), b(std::nan(""));
    a.load_packed_n(x, N);
    b.load_packed_n(y, N);
    double got = a.dot(b);  // = (a*b).sum() with the default (full-width) N
    char buf[64]; std::snprintf(buf, sizeof buf, "vector dot, N=%d", N);
    check(buf, got, want);
  }

  // 2. Square matrix multiply C = A*B with n not a multiple of lanes (GEMM path)
  for (int n : {3, 5, 7, 9, 13}) {
    static double A[32 * 32], B[32 * 32], C[32 * 32];
    for (int i = 0; i < n * n; ++i) { A[i] = (i % 7) + 1; B[i] = (i % 5) + 1; C[i] = 0; }
    using row_t = RAJA::expt::RowIndex<int, mat_t>;
    using col_t = RAJA::expt::ColIndex<int, mat_t>;
    RAJA::View<double, RAJA::Layout<2, int, 1>> Av(A, n, n), Bv(B, n, n), Cv(C, n, n);
    Cv(row_t::all(), col_t::all()) = Av(row_t::all(), col_t::all()) * Bv(row_t::all(), col_t::all());
    double got = 0, want = 0; int nans = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
      double ref = 0; for (int k = 0; k < n; ++k) ref += A[i * n + k] * B[k * n + j];
      want += ref; got += C[i * n + j]; if (std::isnan(C[i * n + j])) nans++;
    }
    char buf[64]; std::snprintf(buf, sizeof buf, "matrix C=A*B, n=%d (NaNs=%d)", n, nans);
    check(buf, got, want);
  }

  // 3. Matrix-vector product y = A*x (GESUMMV path: reduction across lanes)
  for (int n : {3, 5, 7, 9}) {
    static double A[32 * 32], x[32], y[32];
    for (int i = 0; i < n * n; ++i) A[i] = (i % 7) + 1;
    for (int i = 0; i < n; ++i) { x[i] = i + 1; y[i] = 0; }
    using row_t = RAJA::expt::RowIndex<int, mat_t>;
    using col_t = RAJA::expt::ColIndex<int, mat_t>;
    using cvec_t = RAJA::expt::VectorIndex<int, vec_t>;
    RAJA::View<double, RAJA::Layout<2, int, 1>> Av(A, n, n);
    RAJA::View<double, RAJA::Layout<1, int, 0>> Xv(x, n), Yv(y, n);
    Yv(cvec_t::all()) = Av(row_t::all(), col_t::all()) * Xv(cvec_t::all());
    double got = 0, want = 0; int nans = 0;
    for (int i = 0; i < n; ++i) {
      double ref = 0; for (int k = 0; k < n; ++k) ref += A[i * n + k] * x[k];
      want += ref; got += y[i]; if (std::isnan(y[i])) nans++;
    }
    char buf[64]; std::snprintf(buf, sizeof buf, "matvec y=A*x, n=%d (NaNs=%d)", n, nans);
    check(buf, got, want);
  }

  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
