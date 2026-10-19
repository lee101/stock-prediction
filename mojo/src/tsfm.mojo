"""Timestral-209M univariate one-step forward (context 512 log returns -> 21 quantiles) on the GPU.

Tensor-core GEMMs from the Modular kernel library; everything else is custom fused kernels. Residual stream is fp32,
GEMM operands/outputs are DT (bf16 or fp16). Math follows mojo/torch_fast.py, which matches the official model.
"""

from std.math import sqrt, exp, asinh, sinh, log, ceildiv
from std.memory import stack_allocation, unsafe_memcpy
from std.sys import argv, size_of
from std.sys import get_defined_int
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from layout import TileTensor, row_major, Idx
from linalg.matmul.gpu import _matmul_gpu, multistage_gemm
from linalg.utils_gpu import MatmulConfig
from std.utils.index import Index
import linalg.matmul.vendor.blas as vendor_blas

comptime DT = DType.bfloat16
comptime EXT = "bf16" if DT == DType.bfloat16 else "f16"
comptime F32 = DType.float32
comptime D = 768
comptime NH = 12
comptime HD = 64
comptime FF = 2048
comptime L = get_defined_int["L", 16]()
comptime S = L + 1
comptime CTX = L * 32
comptime NL = 24
comptime USE_BLAS = False
comptime DATA = "mojo/data/"

comptime BufD = DeviceBuffer[DT]
comptime BufF = DeviceBuffer[F32]


@always_inline
def mpd(b: BufD) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def mpf(b: BufF) -> Pointer[Float32, MutAnyOrigin]:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def load[dt: DType](ctx: DeviceContext, path: String, n: Int) raises -> DeviceBuffer[dt]:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    var want = n * size_of[Scalar[dt]]()
    if len(bytes) != want:
        raise Error(path + ": expected " + String(want) + " bytes, got " + String(len(bytes)))
    var hb = ctx.enqueue_create_host_buffer[dt](n)
    unsafe_memcpy(dest=hb.unsafe_ptr().bitcast[UInt8](), src=bytes.unsafe_ptr(), count=want)
    var db = ctx.enqueue_create_buffer[dt](n)
    ctx.enqueue_copy(db, hb)
    ctx.synchronize()
    return db^


comptime PD = Pointer[Scalar[DT], MutAnyOrigin]


def mmp[N: Int, K: Int](ctx: DeviceContext, c: PD, a: PD, w: PD, m: Int) raises:
    """c[m, N] = a[m, K] @ w[N, K]^T on raw pointers."""
    var tc = TileTensor(c, row_major(m, Idx[N]))
    var ta = TileTensor(a, row_major(m, Idx[K]))
    var tw = TileTensor(w, row_major(Idx[N], Idx[K]))
    comptime if USE_BLAS:
        vendor_blas.matmul(ctx, tc, ta, tw, c_row_major=True, transpose_b=True)
    else:
        comptime cfg = MatmulConfig[DT, DT, DT, True](
            block_tile_shape=Index(128, 128, 32), warp_tile_shape=Index(64, 64, 32), num_pipeline_stages=3)
        if m >= 2048:
            multistage_gemm[transpose_b=True, config=cfg](tc, ta, tw, ctx)
        else:
            _matmul_gpu[use_tensor_core=True, transpose_b=True](tc, ta, tw, ctx)


def mm[N: Int, K: Int](ctx: DeviceContext, c: BufD, a: BufD, w: BufD, m: Int) raises:
    mmp[N, K](ctx, mpd(c), mpd(a), mpd(w), m)


struct Layer(Movable):
    var variate: Bool
    var n0: BufF
    var qkvg: BufD
    var o0: BufD
    var nm: BufF
    var w12: BufD
    var w3: BufD
    var n1: BufF
    var vg: BufD
    var o1: BufD

    def __init__(out self, ctx: DeviceContext, i: Int, variate: Bool) raises:
        var p = DATA + "l" + String(i) + "_"
        self.variate = variate
        self.n0 = load[F32](ctx, p + "n0.f32", D)
        self.qkvg = load[DT](ctx, p + "qkvg." + EXT, 4 * D * D)
        self.o0 = load[DT](ctx, p + "o0." + EXT, D * D)
        self.nm = load[F32](ctx, p + "nm.f32", D)
        self.w12 = load[DT](ctx, p + "w12." + EXT, 2 * FF * D)
        self.w3 = load[DT](ctx, p + "w3." + EXT, D * FF)
        if variate:
            self.n1 = load[F32](ctx, p + "n1.f32", D)
            self.vg = load[DT](ctx, p + "vg." + EXT, 2 * D * D)
            self.o1 = load[DT](ctx, p + "o1." + EXT, D * D)
        else:
            self.n1 = ctx.enqueue_create_buffer[F32](1)
            self.vg = ctx.enqueue_create_buffer[DT](1)
            self.o1 = ctx.enqueue_create_buffer[DT](1)


struct Weights(Movable):
    var layers: List[Layer]
    var fq: BufF
    var e1b: BufF
    var e2b: BufF
    var erb: BufF
    var on: BufF
    var cosb: BufF
    var sinb: BufF
    var qw: BufF
    var e1: BufD
    var e2: BufD
    var er: BufD
    var h1: BufD
    var h2: BufD
    var hr: BufD

    def __init__(out self, ctx: DeviceContext) raises:
        self.layers = List[Layer]()
        for i in range(NL):
            self.layers.append(Layer(ctx, i, i % 4 == 3))
        self.fq = load[F32](ctx, DATA + "fq.f32", D)
        self.e1b = load[F32](ctx, DATA + "e1b.f32", 4 * D)
        self.e2b = load[F32](ctx, DATA + "e2b.f32", D)
        self.erb = load[F32](ctx, DATA + "erb.f32", D)
        self.on = load[F32](ctx, DATA + "on.f32", D)
        self.cosb = load[F32](ctx, DATA + "cos.f32", 33 * HD)
        self.sinb = load[F32](ctx, DATA + "sin.f32", 33 * HD)
        self.qw = load[F32](ctx, DATA + "qw.f32", 21)
        self.e1 = load[DT](ctx, DATA + "e1." + EXT, 4 * D * 32)
        self.e2 = load[DT](ctx, DATA + "e2." + EXT, D * 4 * D)
        self.er = load[DT](ctx, DATA + "er." + EXT, D * 32)
        self.h1 = load[DT](ctx, DATA + "h1." + EXT, 4 * D * D)
        self.h2 = load[DT](ctx, DATA + "h2." + EXT, 32 * 4 * D)
        self.hr = load[DT](ctx, DATA + "hr." + EXT, 32 * D)


struct Acts(Movable):
    """Activation buffers for R sequences and M token rows (M = R * 17 for the full stack, R for the pruned last layer)."""
    var R: Int
    var M: Int
    var xin: BufF
    var z: BufD
    var loc: BufF
    var sc: BufF
    var e1h: BufD
    var e2o: BufD
    var ero: BufD
    var x: BufF
    var h: BufD
    var big: BufD
    var o: BufD
    var t: BufD
    var u: BufD
    var hh: BufD
    var r2: BufD
    var rr: BufD
    var out: BufF

    def __init__(out self, ctx: DeviceContext, R: Int, M: Int, full: Bool, nout: Int) raises:
        self.R = R
        self.M = M
        var rr_ = R if full else 1
        self.xin = ctx.enqueue_create_buffer[F32](rr_ * CTX)
        self.z = ctx.enqueue_create_buffer[DT](rr_ * L * 32)
        self.loc = ctx.enqueue_create_buffer[F32](rr_ * L)
        self.sc = ctx.enqueue_create_buffer[F32](rr_ * L)
        self.e1h = ctx.enqueue_create_buffer[DT](rr_ * L * 4 * D)
        self.e2o = ctx.enqueue_create_buffer[DT](rr_ * L * D)
        self.ero = ctx.enqueue_create_buffer[DT](rr_ * L * D)
        self.x = ctx.enqueue_create_buffer[F32](M * D)
        self.h = ctx.enqueue_create_buffer[DT](M * D)
        self.big = ctx.enqueue_create_buffer[DT](M * 2 * FF)
        self.o = ctx.enqueue_create_buffer[DT](M * D)
        self.t = ctx.enqueue_create_buffer[DT](M * D)
        self.u = ctx.enqueue_create_buffer[DT](M * FF)
        var rh = R if full else 1
        self.hh = ctx.enqueue_create_buffer[DT](rh * 4 * D)
        self.r2 = ctx.enqueue_create_buffer[DT](rh * 32)
        self.rr = ctx.enqueue_create_buffer[DT](rh * 32)
        self.out = ctx.enqueue_create_buffer[F32](nout * 21 if full else 1)


def k_prep(ctx: DeviceContext, m: Acts, nb: Int) raises:
    """Causal running mean/std (fp64 Welford, as the reference), per-patch loc/scale and asinh-normalized patches.
    Rows >= nb are the sign-flipped copies of rows - nb (symmetric averaging)."""
    var R = m.R
    var px = mpf(m.xin)
    var pz = mpd(m.z)
    var pl = mpf(m.loc)
    var ps = mpf(m.sc)
    var nb64 = Int64(nb)
    var R64 = Int64(R)

    def k() {var px, var pz, var pl, var ps, var R64, var nb64}:
        var R = Int(R64)
        var nb_ = Int(nb64)
        var r = Int(block_idx.x * block_dim.x + thread_idx.x)
        if r >= R:
            return
        var src = r
        var sgn: Float32 = 1.0
        if r >= nb_:
            src = r - nb_
            sgn = -1.0
        var row = px + src * CTX
        var cumx: Float64 = 0
        var mean: Float64 = 0
        var m2: Float64 = 0
        for t in range(CTX):
            var x = Float64(sgn * row[t])
            var prev = mean
            cumx += x
            mean = cumx / Float64(t + 1)
            m2 += (x - prev) * (x - mean)
            if t % 32 == 31:
                var n = t + 1
                var den = Float64(n - 1)
                if den < 1.0:
                    den = 1.0
                var scale = sqrt(m2 / den)
                if scale < 1e-6:
                    scale = 1e-6
                var lf = Float32(mean)
                var sf = Float32(scale)
                var p = t // 32
                pl[r * L + p] = lf
                ps[r * L + p] = sf
                var dn = sf
                if dn < 1.1920929e-07:
                    dn = 1.1920929e-07
                for j in range(32):
                    var v = (sgn * row[p * 32 + j] - lf) / dn
                    pz[(r * L + p) * 32 + j] = Scalar[DT](asinh(v))

    ctx.enqueue_function(k, grid_dim=ceildiv(R, 64), block_dim=64)


def k_silu_bias(ctx: DeviceContext, m: Acts, e1b: BufF, rows: Int) raises:
    var p = mpd(m.e1h)
    var pb = mpf(e1b)
    var tot64 = Int64(rows * 4 * D)

    def k() {var p, var pb, var tot64}:
        var tot = Int(tot64)
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < tot:
            var v = Float32(p[i]) + pb[i % (4 * D)]
            p[i] = Scalar[DT](v / (1.0 + exp(-v)))

    ctx.enqueue_function(k, grid_dim=ceildiv(Int(tot64), 256), block_dim=256)


def k_embed(ctx: DeviceContext, m: Acts, w: Weights, R: Int, n0: BufF) raises:
    """x[b, p] = e2 + b2 + er + erb (p < 16), x[b, 16] = forecast query; h = rmsnorm(x) * n0."""
    var px = mpf(m.x)
    var ph = mpd(m.h)
    var pe = mpd(m.e2o)
    var pr = mpd(m.ero)
    var pb2 = mpf(w.e2b)
    var pbr = mpf(w.erb)
    var pfq = mpf(w.fq)
    var pw = mpf(n0)

    def k() {var px, var ph, var pe, var pr, var pb2, var pbr, var pfq, var pw}:
        var r = Int(block_idx.x)
        var t = Int(thread_idx.x)
        var b = r // S
        var p = r % S
        var s: Float32 = 0
        var v0: Float32 = 0
        var v1: Float32 = 0
        var v2: Float32 = 0
        comptime for i in range(3):
            var idx = t + 256 * i
            var v: Float32
            if p < L:
                var src = (b * L + p) * D + idx
                v = Float32(pe[src]) + pb2[idx] + Float32(pr[src]) + pbr[idx]
            else:
                v = pfq[idx]
            px[r * D + idx] = v
            s += v * v
            comptime if i == 0:
                v0 = v
            elif i == 1:
                v1 = v
            else:
                v2 = v
        var tot = block_sum[block_size=256](s)
        var inv = Float32(1.0) / sqrt(tot / Float32(D) + 1e-6)
        ph[r * D + t] = Scalar[DT](v0 * inv * pw[t])
        ph[r * D + t + 256] = Scalar[DT](v1 * inv * pw[t + 256])
        ph[r * D + t + 512] = Scalar[DT](v2 * inv * pw[t + 512])

    ctx.enqueue_function(k, grid_dim=R * S, block_dim=256)


def k_add_rms(ctx: DeviceContext, m: Acts, M: Int, w: BufF) raises:
    """x += t (fp32 residual); h = rmsnorm(x) * w."""
    var px = mpf(m.x)
    var pt = mpd(m.t)
    var ph = mpd(m.h)
    var pw = mpf(w)

    def k() {var px, var pt, var ph, var pw}:
        var r = Int(block_idx.x)
        var t = Int(thread_idx.x)
        var s: Float32 = 0
        var v0 = px[r * D + t] + Float32(pt[r * D + t])
        var v1 = px[r * D + t + 256] + Float32(pt[r * D + t + 256])
        var v2 = px[r * D + t + 512] + Float32(pt[r * D + t + 512])
        px[r * D + t] = v0
        px[r * D + t + 256] = v1
        px[r * D + t + 512] = v2
        s = v0 * v0 + v1 * v1 + v2 * v2
        var tot = block_sum[block_size=256](s)
        var inv = Float32(1.0) / sqrt(tot / Float32(D) + 1e-6)
        ph[r * D + t] = Scalar[DT](v0 * inv * pw[t])
        ph[r * D + t + 256] = Scalar[DT](v1 * inv * pw[t + 256])
        ph[r * D + t + 512] = Scalar[DT](v2 * inv * pw[t + 512])

    ctx.enqueue_function(k, grid_dim=M, block_dim=256)


def k_attn(ctx: DeviceContext, m: Acts, w: Weights, R: Int) raises:
    """Causal attention over the 17 tokens of each (row, head) with RoPE and the sigmoid gate; reads big = [q|k|v|gate].
    One block per (row, head), 136 threads (17 tokens x 8 sixteen-byte vectors); vectorized global access."""
    var pq = mpd(m.big)
    var po = mpd(m.o)
    var pc = mpf(w.cosb)
    var ps = mpf(w.sinb)
    comptime NT = S * 8
    comptime KS = 65

    def k() {var pq, var po, var pc, var ps}:
        var h = Int(block_idx.x) % NH
        var b = Int(block_idx.x) // NH
        var t = Int(thread_idx.x)
        var qs = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var ks = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var vs = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var sm = stack_allocation[S * S, Float32, address_space=AddressSpace.SHARED]()
        var j8 = t // 8
        var v8 = t % 8
        var rowp = pq + (b * S + j8) * (4 * D) + h * HD
        var vv = (rowp + 2 * D + 8 * v8).unsafe_load[width=8, alignment=16]()
        comptime for i in range(8):
            vs[j8 * KS + 8 * v8 + i] = Float32(vv[i])
        if t < S * 4:
            var j = t // 4
            var u = t % 4
            var row = pq + (b * S + j) * (4 * D) + h * HD
            var qa = (row + 8 * u).unsafe_load[width=8, alignment=16]()
            var qb = (row + 32 + 8 * u).unsafe_load[width=8, alignment=16]()
            var ka = (row + D + 8 * u).unsafe_load[width=8, alignment=16]()
            var kb = (row + D + 32 + 8 * u).unsafe_load[width=8, alignment=16]()
            comptime for i in range(8):
                var c = pc[j * HD + 8 * u + i]
                var s = ps[j * HD + 8 * u + i]
                var qai = Float32(qa[i])
                var qbi = Float32(qb[i])
                var kai = Float32(ka[i])
                var kbi = Float32(kb[i])
                qs[j * KS + 8 * u + i] = qai * c - qbi * s
                qs[j * KS + 32 + 8 * u + i] = qbi * c + qai * s
                ks[j * KS + 8 * u + i] = kai * c - kbi * s
                ks[j * KS + 32 + 8 * u + i] = kbi * c + kai * s
        barrier()
        for idx in range(t, S * S, NT):
            var qi = idx // S
            var kj = idx % S
            if kj <= qi:
                var a: Float32 = 0
                for d in range(HD):
                    a += qs[qi * KS + d] * ks[kj * KS + d]
                sm[idx] = a * 0.125
            else:
                sm[idx] = -3.0e38
        barrier()
        if t < S:
            var mx: Float32 = -3.0e38
            for j in range(t + 1):
                mx = max(mx, sm[t * S + j])
            var tot: Float32 = 0
            for j in range(t + 1):
                var e = exp(sm[t * S + j] - mx)
                sm[t * S + j] = e
                tot += e
            var inv = Float32(1.0) / tot
            for j in range(t + 1):
                sm[t * S + j] = sm[t * S + j] * inv
        barrier()
        var qi = t // 8
        var gv = (pq + (b * S + qi) * (4 * D) + 3 * D + h * HD + 8 * v8).unsafe_load[width=8, alignment=16]()
        var acc = SIMD[DType.float32, 8](0)
        for j in range(qi + 1):
            var p_ = sm[qi * S + j]
            comptime for i in range(8):
                acc[i] += p_ * vs[j * KS + 8 * v8 + i]
        var res = SIMD[DT, 8](0)
        comptime for i in range(8):
            var g = Float32(gv[i])
            res[i] = Scalar[DT](acc[i] / (Float32(1.0) + exp(-g)))
        (po + (b * S + qi) * D + h * HD + 8 * v8).unsafe_store[width=8, alignment=16](res)

    ctx.enqueue_function(k, grid_dim=R * NH, block_dim=NT)


def k_vgate(ctx: DeviceContext, m: Acts, M: Int) raises:
    """o = v * sigmoid(gate) from big = [v | gate] (variate attention over a single key is the identity on v)."""
    var pb = mpd(m.big)
    var po = mpd(m.o)
    var tot64 = Int64(M * D)

    def k() {var pb, var po, var tot64}:
        var tot = Int(tot64)
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < tot:
            var r = i // D
            var c = i % D
            var v = Float32(pb[r * 2 * D + c])
            var g = Float32(pb[r * 2 * D + D + c])
            po[i] = Scalar[DT](v / (Float32(1.0) + exp(-g)))

    ctx.enqueue_function(k, grid_dim=ceildiv(Int(tot64), 256), block_dim=256)


def k_swiglu(ctx: DeviceContext, m: Acts, M: Int) raises:
    var pb = mpd(m.big)
    var pu = mpd(m.u)
    var tot64 = Int64(M * FF)

    def k() {var pb, var pu, var tot64}:
        var tot = Int(tot64)
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < tot:
            var r = i // FF
            var c = i % FF
            var g = Float32(pb[r * 2 * FF + c])
            var u = Float32(pb[r * 2 * FF + FF + c])
            pu[i] = Scalar[DT](g * (u / (Float32(1.0) + exp(-u))))

    ctx.enqueue_function(k, grid_dim=ceildiv(Int(tot64), 256), block_dim=256)


def k_silu(ctx: DeviceContext, m: Acts, n_: Int) raises:
    var p = mpd(m.hh)
    var n64 = Int64(n_)

    def k() {var p, var n64}:
        var n = Int(n64)
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < n:
            var v = Float32(p[i])
            p[i] = Scalar[DT](v / (Float32(1.0) + exp(-v)))

    ctx.enqueue_function(k, grid_dim=ceildiv(n_, 256), block_dim=256)


def k_gather_last(ctx: DeviceContext, a: Acts, l: Acts) raises:
    """Copy the forecast-query token (index 16) of every row into the compact last-layer buffers."""
    var px = mpf(a.x)
    var ph = mpd(a.h)
    var lx = mpf(l.x)
    var lh = mpd(l.h)

    def k() {var px, var ph, var lx, var lh}:
        var b = Int(block_idx.x)
        var t = Int(thread_idx.x)
        comptime for i in range(3):
            var idx = t + 256 * i
            lx[b * D + idx] = px[(b * S + (S - 1)) * D + idx]
            lh[b * D + idx] = ph[(b * S + (S - 1)) * D + idx]

    ctx.enqueue_function(k, grid_dim=a.R, block_dim=256)


def k_attn_last(ctx: DeviceContext, a: Acts, l: Acts, w: Weights, R: Int) raises:
    """Attention of the forecast-query token only: keys/values of all 17 tokens from a.big = [k|v] (stride 1536),
    q and gate from l.big (two contiguous (R, 768) blocks); output to l.o."""
    var pkv = mpd(a.big)
    var pq = mpd(l.big)
    var pg = mpd(l.big) + R * D
    var po = mpd(l.o)
    var pc = mpf(w.cosb)
    var ps = mpf(w.sinb)
    comptime KS = 65

    def k() {var pkv, var pq, var pg, var po, var pc, var ps}:
        var h = Int(block_idx.x) % NH
        var b = Int(block_idx.x) // NH
        var d = Int(thread_idx.x)
        var ks = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var vs = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var qr = stack_allocation[HD, Float32, address_space=AddressSpace.SHARED]()
        var qs = stack_allocation[HD, Float32, address_space=AddressSpace.SHARED]()
        var sm = stack_allocation[64, Float32, address_space=AddressSpace.SHARED]()
        var pd = d + 32
        if d >= 32:
            pd = d - 32
        for j in range(S):
            var row = pkv + (b * S + j) * (2 * D) + h * HD
            var c = pc[j * HD + d]
            var s = ps[j * HD + d]
            var kd = Float32(row[d])
            var kp = Float32(row[pd])
            if d < 32:
                ks[j * KS + d] = kd * c - kp * s
            else:
                ks[j * KS + d] = kd * c + kp * s
            vs[j * KS + d] = Float32(row[D + d])
        qr[d] = Float32(pq[b * D + h * HD + d])
        barrier()
        var c16 = pc[(S - 1) * HD + d]
        var s16 = ps[(S - 1) * HD + d]
        if d < 32:
            qs[d] = qr[d] * c16 - qr[pd] * s16
        else:
            qs[d] = qr[d] * c16 + qr[pd] * s16
        barrier()
        if d < S:
            var a_: Float32 = 0
            for i in range(HD):
                a_ += qs[i] * ks[d * KS + i]
            sm[d] = a_ * 0.125
        barrier()
        if d == 0:
            var mx: Float32 = -3.0e38
            for j in range(S):
                mx = max(mx, sm[j])
            var tot: Float32 = 0
            for j in range(S):
                var e = exp(sm[j] - mx)
                sm[j] = e
                tot += e
            var inv = Float32(1.0) / tot
            for j in range(S):
                sm[j] = sm[j] * inv
        barrier()
        var acc: Float32 = 0
        for j in range(S):
            acc += sm[j] * vs[j * KS + d]
        var g = Float32(pg[b * D + h * HD + d])
        var gs = Float32(1.0) / (Float32(1.0) + exp(-g))
        po[b * D + h * HD + d] = Scalar[DT](acc * gs)

    ctx.enqueue_function(k, grid_dim=R * NH, block_dim=HD)


def quantiles_row(pr2: Pointer[Scalar[DT], MutAnyOrigin], prr: Pointer[Scalar[DT], MutAnyOrigin], pqw: Pointer[Float32, MutAnyOrigin], lf: Float32, sf: Float32, row: Int) -> Array[Float32, 21]:
    var raw = Array[Float32, 21](fill=0)
    var inc = Array[Float32, 21](fill=0)
    for q in range(21):
        var r = Float32(pr2[row * 32 + q]) + Float32(prr[row * 32 + q])
        raw[q] = r
        var sp = r
        if r <= 20.0:
            sp = log(Float32(1.0) + exp(r))
        inc[q] = pqw[q] * sp
    var out = Array[Float32, 21](fill=0)
    var med = raw[10]
    out[10] = med
    var acc: Float32 = 0
    for i in range(11, 21):
        acc += inc[i]
        out[i] = med + acc
    acc = 0
    for i in range(9, -1, -1):
        acc += inc[i]
        out[i] = med - acc
    for q in range(21):
        out[q] = sinh(out[q]) * sf + lf
    return out^


def k_head_post(ctx: DeviceContext, m: Acts, w: Weights, nout: Int, sa: Bool) raises:
    var pr2 = mpd(m.r2)
    var prr = mpd(m.rr)
    var pqw = mpf(w.qw)
    var pl = mpf(m.loc)
    var ps = mpf(m.sc)
    var po = mpf(m.out)
    var sa_ = sa
    var nout64 = Int64(nout)

    def k() {var pr2, var prr, var pqw, var pl, var ps, var po, var nout64, var sa_}:
        var nout = Int(nout64)
        var b = Int(block_idx.x * block_dim.x + thread_idx.x)
        if b >= nout:
            return
        var qa = quantiles_row(pr2, prr, pqw, pl[b * L + L - 1], ps[b * L + L - 1], b)
        if sa_:
            var qb = quantiles_row(pr2, prr, pqw, pl[(b + nout) * L + L - 1], ps[(b + nout) * L + L - 1], b + nout)
            for q in range(21):
                po[b * 21 + q] = 0.5 * (qa[q] - qb[20 - q])
        else:
            for q in range(21):
                po[b * 21 + q] = qa[q]

    ctx.enqueue_function(k, grid_dim=ceildiv(nout, 64), block_dim=64)


struct Prof(Movable):
    var on: Bool
    var acc: List[Float64]
    var last: Int

    def __init__(out self, on: Bool):
        self.on = on
        self.acc = List[Float64]()
        for _ in range(16):
            self.acc.append(0.0)
        self.last = Int(perf_counter_ns())

    def lap(mut self, ctx: DeviceContext, i: Int) raises:
        if self.on:
            ctx.synchronize()
            var n = Int(perf_counter_ns())
            self.acc[i] += Float64(n - self.last)
            self.last = n


def pname(i: Int) -> String:
    if i == 0: return "prep+embed"
    if i == 1: return "qkvg gemm"
    if i == 2: return "attn"
    if i == 3: return "o gemm"
    if i == 4: return "add_rms"
    if i == 5: return "vg gemm"
    if i == 6: return "vgate"
    if i == 7: return "w12 gemm"
    if i == 8: return "swiglu"
    if i == 9: return "w3 gemm"
    return "head"


def forward(ctx: DeviceContext, m: Weights, A: Acts, LA: Acts, nb: Int, sa: Bool, mut pf: Prof) raises:
    var R = A.R
    var M = R * S
    pf.last = Int(perf_counter_ns())
    k_prep(ctx, A, nb)
    mm[4 * D, 32](ctx, A.e1h, A.z, m.e1, R * L)
    k_silu_bias(ctx, A, m.e1b, R * L)
    mm[D, 4 * D](ctx, A.e2o, A.e1h, m.e2, R * L)
    mm[D, 32](ctx, A.ero, A.z, m.er, R * L)
    k_embed(ctx, A, m, R, m.layers[0].n0)
    pf.lap(ctx, 0)
    for i in range(NL - 1):
        ref ly = m.layers[i]
        mm[4 * D, D](ctx, A.big, A.h, ly.qkvg, M)
        pf.lap(ctx, 1)
        k_attn(ctx, A, m, R)
        pf.lap(ctx, 2)
        mm[D, D](ctx, A.t, A.o, ly.o0, M)
        pf.lap(ctx, 3)
        if ly.variate:
            k_add_rms(ctx, A, M, ly.n1)
            pf.lap(ctx, 4)
            mm[2 * D, D](ctx, A.big, A.h, ly.vg, M)
            pf.lap(ctx, 5)
            k_vgate(ctx, A, M)
            pf.lap(ctx, 6)
            mm[D, D](ctx, A.t, A.o, ly.o1, M)
            pf.lap(ctx, 3)
        k_add_rms(ctx, A, M, ly.nm)
        pf.lap(ctx, 4)
        mm[2 * FF, D](ctx, A.big, A.h, ly.w12, M)
        pf.lap(ctx, 7)
        k_swiglu(ctx, A, M)
        pf.lap(ctx, 8)
        mm[D, FF](ctx, A.t, A.u, ly.w3, M)
        pf.lap(ctx, 9)
        k_add_rms(ctx, A, M, m.layers[i + 1].n0)
        pf.lap(ctx, 4)
    # last layer: only the forecast-query token's output is used, so everything after K/V runs on R rows
    ref ly = m.layers[NL - 1]
    mmp[2 * D, D](ctx, mpd(A.big), mpd(A.h), mpd(ly.qkvg) + D * D, M)
    k_gather_last(ctx, A, LA)
    mmp[D, D](ctx, mpd(LA.big), mpd(LA.h), mpd(ly.qkvg), R)
    mmp[D, D](ctx, mpd(LA.big) + R * D, mpd(LA.h), mpd(ly.qkvg) + 3 * D * D, R)
    k_attn_last(ctx, A, LA, m, R)
    mm[D, D](ctx, LA.t, LA.o, ly.o0, R)
    k_add_rms(ctx, LA, R, ly.n1)
    mm[2 * D, D](ctx, LA.big, LA.h, ly.vg, R)
    k_vgate(ctx, LA, R)
    mm[D, D](ctx, LA.t, LA.o, ly.o1, R)
    k_add_rms(ctx, LA, R, ly.nm)
    mm[2 * FF, D](ctx, LA.big, LA.h, ly.w12, R)
    k_swiglu(ctx, LA, R)
    mm[D, FF](ctx, LA.t, LA.u, ly.w3, R)
    k_add_rms(ctx, LA, R, m.on)
    pf.lap(ctx, 4)
    mm[4 * D, D](ctx, A.hh, LA.h, m.h1, R)
    k_silu(ctx, A, R * 4 * D)
    mm[32, 4 * D](ctx, A.r2, A.hh, m.h2, R)
    mm[32, D](ctx, A.rr, LA.h, m.hr, R)
    k_head_post(ctx, A, m, nb, sa)
    pf.lap(ctx, 10)


def run_file(sa: Bool, inpath: String, outpath: String, n: Int, ch: Int) raises:
    """Forecast n windows of CTX log returns from inpath (float32 [n, CTX]) into outpath (float32 [n, 21])."""
    var R = ch * 2 if sa else ch
    with DeviceContext() as ctx:
        var w = Weights(ctx)
        var A = Acts(ctx, R, R * S, True, ch)
        var Lt = Acts(ctx, R, R, False, ch)
        var xin = load[F32](ctx, inpath, n * CTX)
        var out_h = ctx.enqueue_create_host_buffer[F32](n * 21)
        var pf0 = Prof(False)
        var done = 0
        while done < n:
            var cnt = min(ch, n - done)
            var src = mpf(xin) + done * CTX
            var dst = mpf(A.xin)
            var total = Int64(cnt * CTX)

            def kc() {var src, var dst, var total}:
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < Int(total):
                    dst[i] = src[i]

            ctx.enqueue_function(kc, grid_dim=ceildiv(Int(total), 256), block_dim=256)
            forward(ctx, w, A, Lt, ch, sa, pf0)
            ctx.synchronize()
            var hb = ctx.enqueue_create_host_buffer[F32](ch * 21)
            ctx.enqueue_copy(hb, A.out)
            ctx.synchronize()
            for i in range(cnt * 21):
                out_h[done * 21 + i] = hb[i]
            done += cnt
        var b = List[UInt8](length=n * 21 * 4, fill=0)
        var pb = b.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(n * 21):
            pb[i] = out_h[i]
        var f = open(outpath, "w")
        f.write_bytes(Span(b))
        f.close()


def main() raises:
    var args = argv()
    if len(args) > 1 and args[1] == "run":
        run_file(args[2] == "1", String(args[3]), String(args[4]), Int(args[5]), Int(args[6]))
        return
    var B = 256
    var sa = False
    var iters = 10
    if len(args) > 1:
        B = Int(args[1])
    if len(args) > 2:
        sa = args[2] == "1"
    if len(args) > 3:
        iters = Int(args[3])
    var R = B * 2 if sa else B
    with DeviceContext() as ctx:
        var w = Weights(ctx)
        var A = Acts(ctx, R, R * S, True, B)
        var Lt = Acts(ctx, R, R, False, B)
        var xin = load[F32](ctx, DATA + "test_in_" + String(CTX) + ".f32", 1024 * CTX)
        ctx.enqueue_copy(A.xin, xin)   # first R*512 floats; mirrored rows are derived in k_prep
        ctx.synchronize()
        var pf0 = Prof(False)
        forward(ctx, w, A, Lt, B, sa, pf0)
        ctx.synchronize()
        var host = ctx.enqueue_create_host_buffer[F32](B * 21)
        ctx.enqueue_copy(host, A.out)
        ctx.synchronize()
        var ref_all = load[F32](ctx, DATA + "test_ref_" + String(CTX) + "_sa" + String(1 if sa else 0) + ".f32", 1024 * 21)
        var rh = ctx.enqueue_create_host_buffer[F32](1024 * 21)
        ctx.enqueue_copy(rh, ref_all)
        ctx.synchronize()
        var maxd: Float64 = 0
        var se: Float64 = 0
        for b in range(B):
            for q in range(21):
                var dd = Float64(host[b * 21 + q] - rh[b * 21 + q])
                if dd < 0:
                    dd = -dd
                if dd > maxd:
                    maxd = dd
            var de = Float64(host[b * 21 + 10] - rh[b * 21 + 10])
            se += de * de
        print("B", B, "sa", sa, "max|dq|", maxd, "median err rms", sqrt(se / Float64(B)))
        var t0 = perf_counter_ns()
        for _ in range(iters):
            forward(ctx, w, A, Lt, B, sa, pf0)
        ctx.synchronize()
        var dt = Float64(perf_counter_ns() - t0) / Float64(iters)
        print("us/window", dt / 1e3 / Float64(B), " ms/batch", dt / 1e6)
        var pf = Prof(True)
        for _ in range(iters):
            forward(ctx, w, A, Lt, B, sa, pf)
        var tot: Float64 = 0
        for i in range(11):
            tot += pf.acc[i]
        for i in range(11):
            print("  ", pname(i), "us/win", pf.acc[i] / Float64(iters) / 1e3 / Float64(B), "pct", 100.0 * pf.acc[i] / tot)
