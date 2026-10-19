from std.time import perf_counter_ns
from std.utils.index import Index
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu, multistage_gemm
from linalg.utils_gpu import MatmulConfig

comptime DT = DType.float16


def run[N: Int, K: Int, BM: Int, BN: Int, BK: Int, WM: Int, WN: Int, ST: Int](ctx: DeviceContext, m: Int) raises -> Float64:
    var a_buf = ctx.enqueue_create_buffer[DT](m * K)
    var b_buf = ctx.enqueue_create_buffer[DT](N * K)
    var c_buf = ctx.enqueue_create_buffer[DT](m * N)
    ctx.enqueue_memset(a_buf, 0)
    ctx.enqueue_memset(b_buf, 0)
    var a = TileTensor(a_buf, row_major(m, Idx[K]))
    var b = TileTensor(b_buf, row_major(Idx[N], Idx[K]))
    var c = TileTensor(c_buf, row_major(m, Idx[N]))
    comptime cfg = MatmulConfig[DT, DT, DT, True](
        block_tile_shape=Index(BM, BN, BK), warp_tile_shape=Index(WM, WN, BK), num_pipeline_stages=ST)
    for _ in range(3):
        multistage_gemm[transpose_b=True, config=cfg](c, a, b, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(20):
        multistage_gemm[transpose_b=True, config=cfg](c, a, b, ctx)
    ctx.synchronize()
    var dtns = Float64(perf_counter_ns() - t0) / 20.0
    return 2.0 * Float64(m) * Float64(N) * Float64(K) / dtns / 1e3


def base[N: Int, K: Int](ctx: DeviceContext, m: Int) raises -> Float64:
    var a_buf = ctx.enqueue_create_buffer[DT](m * K)
    var b_buf = ctx.enqueue_create_buffer[DT](N * K)
    var c_buf = ctx.enqueue_create_buffer[DT](m * N)
    ctx.enqueue_memset(a_buf, 0)
    ctx.enqueue_memset(b_buf, 0)
    var a = TileTensor(a_buf, row_major(m, Idx[K]))
    var b = TileTensor(b_buf, row_major(Idx[N], Idx[K]))
    var c = TileTensor(c_buf, row_major(m, Idx[N]))
    for _ in range(3):
        _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(20):
        _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
    ctx.synchronize()
    var dtns = Float64(perf_counter_ns() - t0) / 20.0
    return 2.0 * Float64(m) * Float64(N) * Float64(K) / dtns / 1e3


def sweep[N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    print("== N", N, "K", K, "M", m, " default", base[N, K](ctx, m))
    print(" 128x64x32 w64x32 s3", run[N, K, 128, 64, 32, 64, 32, 3](ctx, m))
    print(" 128x64x32 w64x32 s4", run[N, K, 128, 64, 32, 64, 32, 4](ctx, m))
    print(" 128x64x32 w64x32 s5", run[N, K, 128, 64, 32, 64, 32, 5](ctx, m))
    print(" 128x64x32 w64x32 s6", run[N, K, 128, 64, 32, 64, 32, 6](ctx, m))
    print(" 128x64x64 w64x32 s3", run[N, K, 128, 64, 64, 64, 32, 3](ctx, m))
    print(" 128x64x64 w64x32 s4", run[N, K, 128, 64, 64, 64, 32, 4](ctx, m))
    print(" 64x64x32 w32x32 s4", run[N, K, 64, 64, 32, 32, 32, 4](ctx, m))
    print(" 64x64x32 w32x32 s6", run[N, K, 64, 64, 32, 32, 32, 6](ctx, m))
    print(" 256x64x32 w64x32 s4", run[N, K, 256, 64, 32, 64, 32, 4](ctx, m))
    print(" 128x128x32 w64x64 s3", run[N, K, 128, 128, 32, 64, 64, 3](ctx, m))
    print(" 128x128x32 w32x64 s3", run[N, K, 128, 128, 32, 32, 64, 3](ctx, m))
    print(" 128x96x32 w64x48 s4", run[N, K, 128, 96, 32, 64, 48, 4](ctx, m))


def main() raises:
    with DeviceContext() as ctx:
        var m = 17408
        sweep[3072, 768](ctx, m)
        sweep[768, 768](ctx, m)
        sweep[4096, 768](ctx, m)
        sweep[768, 2048](ctx, m)
        sweep[1536, 768](ctx, m)
