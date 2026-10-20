from std.time import perf_counter_ns
from std.utils.index import Index
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu import multistage_gemm
from linalg.utils_gpu import MatmulConfig

comptime DT = DType.bfloat16


def one[N: Int, K: Int, TB: Bool, BM: Int, BN: Int, BK: Int, WM: Int, WN: Int, ST: Int](ctx: DeviceContext, m: Int) raises -> Float64:
    comptime AR = K if TB else N          # reduction extent
    comptime CC = N if TB else K          # output columns
    var a_buf = ctx.enqueue_create_buffer[DT](m * AR)
    var w_buf = ctx.enqueue_create_buffer[DT](N * K)
    var c_buf = ctx.enqueue_create_buffer[DT](m * CC)
    ctx.enqueue_memset(a_buf, 0)
    ctx.enqueue_memset(w_buf, 0)
    var ta = TileTensor(a_buf, row_major(m, Idx[AR]))
    var tw = TileTensor(w_buf, row_major(Idx[N], Idx[K]))
    var tc = TileTensor(c_buf, row_major(m, Idx[CC]))
    comptime cfg = MatmulConfig[DT, DT, DT, TB](block_tile_shape=Index(BM, BN, BK), warp_tile_shape=Index(WM, WN, BK), num_pipeline_stages=ST)
    var best: Float64 = 1.0e30
    for _ in range(25):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        multistage_gemm[transpose_b=TB, config=cfg](tc, ta, tw, ctx)
        ctx.synchronize()
        best = min(best, Float64(perf_counter_ns() - t0))
    return 2.0 * Float64(m) * Float64(N) * Float64(K) / best / 1e3


def sweep[N: Int, K: Int, TB: Bool](ctx: DeviceContext, m: Int) raises:
    var r0 = one[N, K, TB, 128, 128, 32, 64, 64, 3](ctx, m)
    var r1 = one[N, K, TB, 128, 64, 32, 64, 32, 4](ctx, m)
    var r2 = one[N, K, TB, 128, 64, 32, 64, 32, 3](ctx, m)
    var r3 = one[N, K, TB, 128, 128, 32, 32, 64, 3](ctx, m)
    var r4 = one[N, K, TB, 256, 64, 32, 64, 32, 4](ctx, m)
    var r5 = one[N, K, TB, 128, 128, 64, 64, 64, 3](ctx, m)
    var r6 = one[N, K, TB, 64, 128, 32, 32, 64, 4](ctx, m)
    var r7 = one[N, K, TB, 128, 128, 32, 64, 64, 4](ctx, m)
    print("NT" if TB else "NN", "N", N, "K", K, "| 128x128x32/3:", Int(r0), " 128x64/4:", Int(r1), " 128x64/3:", Int(r2), " 128x128 w32x64/3:", Int(r3), " 256x64/4:", Int(r4), " 128x128x64/3:", Int(r5), " 64x128/4:", Int(r6), " 128x128/4:", Int(r7))


def main() raises:
    with DeviceContext() as ctx:
        var m = 4352
        sweep[3072, 768, True](ctx, m)
        sweep[768, 768, True](ctx, m)
        sweep[4096, 768, True](ctx, m)
        sweep[768, 2048, True](ctx, m)
        sweep[1536, 768, True](ctx, m)
        sweep[3072, 768, False](ctx, m)
        sweep[768, 768, False](ctx, m)
        sweep[4096, 768, False](ctx, m)
        sweep[768, 2048, False](ctx, m)
        sweep[1536, 768, False](ctx, m)
