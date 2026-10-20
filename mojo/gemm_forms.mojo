from std.time import perf_counter_ns
from std.utils.index import Index
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
import linalg.matmul.vendor.blas as vb
from linalg.matmul.gpu import multistage_gemm
from linalg.utils_gpu import MatmulConfig

comptime DT = DType.bfloat16


def tmin(ctx: DeviceContext, n: Int) -> Float64:
    return 0.0


def forms[N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    var y = ctx.enqueue_create_buffer[DT](m * N)
    var x = ctx.enqueue_create_buffer[DT](m * K)
    var w = ctx.enqueue_create_buffer[DT](N * K)
    var o_nt = ctx.enqueue_create_buffer[DT](m * N)
    var o_nn = ctx.enqueue_create_buffer[DT](m * K)
    var dw16 = ctx.enqueue_create_buffer[DT](N * K)
    var dw32 = ctx.enqueue_create_buffer[DType.float32](N * K)
    ctx.enqueue_memset(y, 0)
    ctx.enqueue_memset(x, 0)
    ctx.enqueue_memset(w, 0)
    var ty = TileTensor(y, row_major(m, Idx[N]))
    var tx = TileTensor(x, row_major(m, Idx[K]))
    var tw = TileTensor(w, row_major(Idx[N], Idx[K]))
    var tont = TileTensor(o_nt, row_major(m, Idx[N]))
    var tonn = TileTensor(o_nn, row_major(m, Idx[K]))
    var tdw16 = TileTensor(dw16, row_major(Idx[N], Idx[K]))
    var tdw32 = TileTensor(dw32, row_major(Idx[N], Idx[K]))
    comptime cfg_nt = MatmulConfig[DT, DT, DT, True](block_tile_shape=Index(128, 128, 32), warp_tile_shape=Index(64, 64, 32), num_pipeline_stages=3)
    comptime cfg_nn = MatmulConfig[DT, DT, DT, False](block_tile_shape=Index(128, 128, 32), warp_tile_shape=Index(64, 64, 32), num_pipeline_stages=3)
    var fl = 2.0 * Float64(m) * Float64(N) * Float64(K)
    var best = Array[Float64, 6](fill=1.0e30)
    for it in range(40):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        vb.matmul(ctx, tont, tx, tw, c_row_major=True, transpose_b=True)
        ctx.synchronize()
        best[0] = min(best[0], Float64(perf_counter_ns() - t0))
        t0 = perf_counter_ns()
        multistage_gemm[transpose_b=True, config=cfg_nt](tont, tx, tw, ctx)
        ctx.synchronize()
        best[1] = min(best[1], Float64(perf_counter_ns() - t0))
        t0 = perf_counter_ns()
        vb.matmul(ctx, tonn, ty, tw, c_row_major=True)
        ctx.synchronize()
        best[2] = min(best[2], Float64(perf_counter_ns() - t0))
        t0 = perf_counter_ns()
        multistage_gemm[transpose_b=False, config=cfg_nn](tonn, ty, tw, ctx)
        ctx.synchronize()
        best[3] = min(best[3], Float64(perf_counter_ns() - t0))
        t0 = perf_counter_ns()
        vb.matmul(ctx, tdw16, ty, tx, c_row_major=True, transpose_a=True)
        ctx.synchronize()
        best[4] = min(best[4], Float64(perf_counter_ns() - t0))
        t0 = perf_counter_ns()
        vb.matmul(ctx, tdw32, ty, tx, c_row_major=True, transpose_a=True)
        ctx.synchronize()
        best[5] = min(best[5], Float64(perf_counter_ns() - t0))
    print("N", N, "K", K, "TFLOPS: NT cublas", fl / best[0] / 1e3, "NT mojo", fl / best[1] / 1e3, "| NN cublas", fl / best[2] / 1e3, "NN mojo", fl / best[3] / 1e3, "| TN bf16-out", fl / best[4] / 1e3, "TN fp32-out", fl / best[5] / 1e3)


def main() raises:
    with DeviceContext() as ctx:
        var m = 4352
        forms[3072, 768](ctx, m)
        forms[768, 768](ctx, m)
        forms[4096, 768](ctx, m)
        forms[768, 2048](ctx, m)
        forms[1536, 768](ctx, m)
