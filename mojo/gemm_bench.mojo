from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu


def bench[dt: DType, N: Int, K: Int](ctx: DeviceContext, m: Int, iters: Int) raises:
    var a_buf = ctx.enqueue_create_buffer[dt](m * K)
    var b_buf = ctx.enqueue_create_buffer[dt](N * K)
    var c_buf = ctx.enqueue_create_buffer[dt](m * N)
    ctx.enqueue_memset(a_buf, 0)
    ctx.enqueue_memset(b_buf, 0)
    var a = TileTensor(a_buf, row_major(m, Idx[K]))
    var b = TileTensor(b_buf, row_major(Idx[N], Idx[K]))
    var c = TileTensor(c_buf, row_major(m, Idx[N]))
    for _ in range(3):
        _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
    ctx.synchronize()
    var dtns = Float64(perf_counter_ns() - t0) / Float64(iters)
    var tf = 2.0 * Float64(m) * Float64(N) * Float64(K) / dtns / 1e3
    print(dt, "M", m, "N", N, "K", K, "us", dtns / 1e3, "TFLOPS", tf)


def main() raises:
    with DeviceContext() as ctx:
        var m = 4352
        comptime for d in [DType.float16, DType.bfloat16]:
            bench[d, 2304, 768](ctx, m, 50)
            bench[d, 768, 768](ctx, m, 50)
            bench[d, 4096, 768](ctx, m, 50)
            bench[d, 768, 2048](ctx, m, 50)
        bench[DType.float32, 4096, 768](ctx, m, 50)
