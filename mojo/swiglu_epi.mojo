from std.time import perf_counter_ns
from std.math import exp
from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu

comptime DT = DType.float16


def main() raises:
    comptime N = 4096
    comptime K = 768
    comptime FF = 2048
    var m = 17408
    with DeviceContext() as ctx:
        var a_buf = ctx.enqueue_create_buffer[DT](m * K)
        var b_buf = ctx.enqueue_create_buffer[DT](N * K)
        var c_buf = ctx.enqueue_create_buffer[DT](m * N)
        var u_buf = ctx.enqueue_create_buffer[DT](m * FF)
        ctx.enqueue_memset(a_buf, 0)
        ctx.enqueue_memset(b_buf, 0)
        var a = TileTensor(a_buf, row_major(m, Idx[K]))
        var b = TileTensor(b_buf, row_major(Idx[N], Idx[K]))
        var c = TileTensor(c_buf, row_major(m, Idx[N]))
        var pu = u_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()

        def swi[_dtype: DType, width: SIMDLength, *, alignment: Int](idx: Coord, val: SIMD[_dtype, width]) {var pu}:
            var row = Int(idx[0])
            var col = Int(idx[1])
            comptime for i in range(width // 2):
                var g = val[2 * i].cast[DType.float32]()
                var u = val[2 * i + 1].cast[DType.float32]()
                pu[row * FF + col // 2 + i] = (g * (u / (1.0 + exp(-u)))).cast[DT]()

        for _ in range(3):
            _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, swi, ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(20):
            _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, swi, ctx)
        ctx.synchronize()
        var dtns = Float64(perf_counter_ns() - t0) / 20.0
        print("epilogue-fused swiglu GEMM us", dtns / 1e3, "TFLOPS", 2.0 * Float64(m) * Float64(N) * Float64(K) / dtns / 1e3)
        t0 = perf_counter_ns()
        for _ in range(20):
            _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
        ctx.synchronize()
        dtns = Float64(perf_counter_ns() - t0) / 20.0
        print("plain GEMM us", dtns / 1e3, "TFLOPS", 2.0 * Float64(m) * Float64(N) * Float64(K) / dtns / 1e3)
