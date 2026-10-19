from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu


def run[N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    comptime dt = DType.float16
    var a_buf = ctx.enqueue_create_buffer[dt](m * K)
    var b_buf = ctx.enqueue_create_buffer[dt](N * K)
    var c_buf = ctx.enqueue_create_buffer[DType.float32](m * N)
    ctx.enqueue_memset(a_buf, 0)
    ctx.enqueue_memset(b_buf, 0)
    ctx.enqueue_memset(c_buf, 0)
    var a = TileTensor(a_buf, row_major(m, Idx[K]))
    var b = TileTensor(b_buf, row_major(Idx[N], Idx[K]))
    var c = TileTensor(c_buf, row_major(m, Idx[N]))
    var r_buf = ctx.enqueue_create_buffer[DType.float32](m * N)
    ctx.enqueue_memset(r_buf, 0)
    var rt = TileTensor(r_buf, row_major(m, Idx[N]))

    def add_res[_dtype: DType, width: SIMDLength, *, alignment: Int](idx: Coord, val: SIMD[_dtype, width]) {var rt} -> SIMD[_dtype, width]:
        var r = rt.load[width=width](idx)
        return val + rebind[SIMD[_dtype, width]](r)

    _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, add_res, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(50):
        _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, add_res, ctx)
    ctx.synchronize()
    var dtns = Float64(perf_counter_ns() - t0) / 50.0
    print("fp16->fp32 +res  M", m, "N", N, "K", K, "us", dtns / 1e3, "TFLOPS", 2.0 * Float64(m) * Float64(N) * Float64(K) / dtns / 1e3)


def main() raises:
    with DeviceContext() as ctx:
        run[768, 768](ctx, 4352)
        run[768, 2048](ctx, 4352)
