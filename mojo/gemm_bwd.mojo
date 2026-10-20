from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
import linalg.matmul.vendor.blas as vb

comptime DT = DType.bfloat16


def bench[N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    var y = ctx.enqueue_create_buffer[DT](m * N)     # dY  (m, N)
    var x = ctx.enqueue_create_buffer[DT](m * K)     # X   (m, K)
    var w = ctx.enqueue_create_buffer[DT](N * K)     # W   (N, K)
    var dx = ctx.enqueue_create_buffer[DT](m * K)    # dX  (m, K) bf16
    var dw = ctx.enqueue_create_buffer[DType.float32](N * K)   # dW (N, K) fp32
    ctx.enqueue_memset(y, 0)
    ctx.enqueue_memset(x, 0)
    ctx.enqueue_memset(w, 0)
    ctx.enqueue_memset(dw, 0)
    var ty = TileTensor(y, row_major(m, Idx[N]))
    var tx = TileTensor(x, row_major(m, Idx[K]))
    var tw = TileTensor(w, row_major(Idx[N], Idx[K]))
    var tdx = TileTensor(dx, row_major(m, Idx[K]))
    var tdw = TileTensor(dw, row_major(Idx[N], Idx[K]))
    var fl = 2.0 * Float64(m) * Float64(N) * Float64(K)
    for _ in range(3):
        vb.matmul(ctx, tdx, ty, tw, c_row_major=True)
        vb.matmul(ctx, tdw, ty, tx, c_row_major=True, transpose_a=True, beta=1.0)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(20):
        vb.matmul(ctx, tdx, ty, tw, c_row_major=True)
    ctx.synchronize()
    var a = Float64(perf_counter_ns() - t0) / 20.0
    t0 = perf_counter_ns()
    for _ in range(20):
        vb.matmul(ctx, tdw, ty, tx, c_row_major=True, transpose_a=True, beta=1.0)
    ctx.synchronize()
    var b = Float64(perf_counter_ns() - t0) / 20.0
    print("N", N, "K", K, "dX bf16 us", a / 1e3, "TFLOPS", fl / a / 1e3, " | dW fp32-accum us", b / 1e3, "TFLOPS", fl / b / 1e3)


def main() raises:
    with DeviceContext() as ctx:
        var m = 8704
        bench[3072, 768](ctx, m)
        bench[768, 768](ctx, m)
        bench[4096, 768](ctx, m)
        bench[768, 2048](ctx, m)
