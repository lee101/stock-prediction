"""Timestral-209M fine-tuning in Mojo: forward with saved activations, hand-written backward, fused AdamW.

Univariate model: CTX log returns -> 32-step x 21-quantile head on the forecast-query token, pinball loss in denormalised
log-return space (mojo/train_ref.py is the autograd oracle). DT = compute dtype of GEMM operands/outputs (fp32 for
validation, bf16/fp16 for speed); master weights, gradients, Adam state, the residual stream, norms and softmax are fp32.
"""

from std.math import sqrt, exp, asinh, sinh, cosh, log, cos, ceildiv
from std.memory import stack_allocation, unsafe_memcpy
from std.sys import argv, size_of, get_defined_int
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from layout import TileTensor, row_major, Idx
import linalg.matmul.vendor.blas as vb
from linalg.matmul.gpu import multistage_gemm
from linalg.utils_gpu import MatmulConfig
from std.utils.index import Index

comptime DT = DType.bfloat16
comptime F32 = DType.float32
comptime D = 768
comptime NH = 12
comptime HD = 64
comptime FF = 2048
comptime L = get_defined_int["L", 16]()
comptime S = L + 1
comptime CTX = L * 32
comptime NL = 24
comptime NQ = 21
comptime NHEAD = 32 * NQ
comptime NG = 128          # blocks for per-column reductions
comptime DATA = "mojo/data/train/"

comptime BufD = DeviceBuffer[DT]
comptime BufF = DeviceBuffer[F32]
comptime PD = Pointer[Scalar[DT], MutAnyOrigin]
comptime PF = Pointer[Float32, MutAnyOrigin]


@always_inline
def pd(b: BufD) -> PD:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def pf(b: BufF) -> PF:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def load_f32(ctx: DeviceContext, path: String, n: Int) raises -> BufF:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    if len(bytes) != n * 4:
        raise Error(path + ": expected " + String(n * 4) + " bytes, got " + String(len(bytes)))
    var hb = ctx.enqueue_create_host_buffer[F32](n)
    unsafe_memcpy(dest=hb.unsafe_ptr().bitcast[UInt8](), src=bytes.unsafe_ptr(), count=n * 4)
    var db = ctx.enqueue_create_buffer[F32](n)
    ctx.enqueue_copy(db, hb)
    ctx.synchronize()
    return db^


def zeros_f(ctx: DeviceContext, n: Int) raises -> BufF:
    var b = ctx.enqueue_create_buffer[F32](n)
    ctx.enqueue_memset(b, 0)
    return b^


def zeros_d(ctx: DeviceContext, n: Int) raises -> BufD:
    var b = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_memset(b, 0)
    return b^


# ---------------------------------------------------------------- GEMMs
# fwd (NT) and input-gradient (NN) products use the tuned Modular multistage tensor-core kernel for half dtypes when the
# problem is large; everything else (fp32 validation, small M, the K=32 embedding, weight gradients) goes through cuBLAS.
comptime LIB = DT != DType.float32


def gnt[N: Int, K: Int](ctx: DeviceContext, c: PD, a: PD, w: PD, m: Int) raises:
    """c[m, N] = a[m, K] @ w[N, K]^T."""
    var tc = TileTensor(c, row_major(m, Idx[N]))
    var ta = TileTensor(a, row_major(m, Idx[K]))
    var tw = TileTensor(w, row_major(Idx[N], Idx[K]))
    comptime if LIB and K >= 256 and N % 128 == 0:
        if m >= 2048:
            comptime if N >= 1536:
                comptime cfg = MatmulConfig[DT, DT, DT, True](
                    block_tile_shape=Index(128, 128, 32), warp_tile_shape=Index(64, 64, 32), num_pipeline_stages=3)
                multistage_gemm[transpose_b=True, config=cfg](tc, ta, tw, ctx)
            else:
                comptime cfg = MatmulConfig[DT, DT, DT, True](
                    block_tile_shape=Index(128, 64, 32), warp_tile_shape=Index(64, 32, 32), num_pipeline_stages=4)
                multistage_gemm[transpose_b=True, config=cfg](tc, ta, tw, ctx)
            return
    vb.matmul(ctx, tc, ta, tw, c_row_major=True, transpose_b=True)


def gnn[N: Int, K: Int](ctx: DeviceContext, c: PD, a: PD, w: PD, m: Int, beta: Float32 = 0.0) raises:
    """c[m, K] (+)= a[m, N] @ w[N, K]   (input gradient)."""
    var tc = TileTensor(c, row_major(m, Idx[K]))
    var ta = TileTensor(a, row_major(m, Idx[N]))
    var tw = TileTensor(w, row_major(Idx[N], Idx[K]))
    comptime if LIB and K >= 256 and K % 128 == 0:
        if m >= 2048 and beta == 0.0:
            comptime cfg = MatmulConfig[DT, DT, DT, False](
                block_tile_shape=Index(128, 64, 32), warp_tile_shape=Index(64, 32, 32), num_pipeline_stages=4)
            multistage_gemm[transpose_b=False, config=cfg](tc, ta, tw, ctx)
            return
    vb.matmul(ctx, tc, ta, tw, c_row_major=True, beta=beta)


def gtn[N: Int, K: Int](ctx: DeviceContext, dw: PF, dy: PD, x: PD, m: Int) raises:
    """dw[N, K] = dy[m, N]^T @ x[m, K]   (weight gradient, fp32 output, overwrites)."""
    var tdw = TileTensor(dw, row_major(Idx[N], Idx[K]))
    var ty = TileTensor(dy, row_major(m, Idx[N]))
    var tx = TileTensor(x, row_major(m, Idx[K]))
    vb.matmul(ctx, tdw, ty, tx, c_row_major=True, transpose_a=True)


# ---------------------------------------------------------------- parameters
struct Param(Movable):
    var name: String
    var n: Int
    var wd: Bool
    var off: Int
    var w: BufF        # fp32 master (view into the flat buffer)
    var g: BufF        # fp32 gradient
    var m: BufF
    var v: BufF
    var c: BufD        # DT compute copy

    def __init__(out self, ctx: DeviceContext, name: String, n: Int, wd: Bool, off: Int, wf: BufF, gf: BufF, mf: BufF, vf: BufF, cf: BufD) raises:
        self.name = name
        self.n = n
        self.wd = wd
        self.off = off
        self.w = wf.create_sub_buffer[F32](off, n)
        self.g = gf.create_sub_buffer[F32](off, n)
        self.m = mf.create_sub_buffer[F32](off, n)
        self.v = vf.create_sub_buffer[F32](off, n)
        self.c = cf.create_sub_buffer[DT](off, n)
        var src = load_f32(ctx, DATA + name + ".f32", n)
        ctx.enqueue_copy(self.w, src)
        ctx.synchronize()


def k_cast_f2d(ctx: DeviceContext, src: BufF, dst: BufD, n: Int) raises:
    var ps = pf(src)
    var po = pd(dst)
    var n64 = Int64(n)

    def k() {var ps, var po, var n64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(n64):
            po[i] = Scalar[DT](ps[i])

    ctx.enqueue_function(k, grid_dim=ceildiv(n, 256), block_dim=256)


# global parameter indices
comptime P_FQ = 0
comptime P_E1 = 1
comptime P_E1B = 2
comptime P_E2 = 3
comptime P_E2B = 4
comptime P_ER = 5
comptime P_ERB = 6
comptime P_ON = 7
comptime P_H1 = 8
comptime P_H2 = 9
comptime P_HR = 10
# per-layer offsets
comptime O_N0 = 0
comptime O_QKVG = 1
comptime O_O0 = 2
comptime O_NM = 3
comptime O_W12 = 4
comptime O_W3 = 5
comptime O_N1 = 6
comptime O_VG = 7
comptime O_O1 = 8


def is_var(l: Int) -> Bool:
    return l % 4 == 3


def lbase(l: Int) -> Int:
    var nvar = l // 4
    return 11 + 6 * l + 3 * nvar


def spec_n(i: Int) -> Int:
    return 0


struct Net(Movable):
    var P: List[Param]
    var cosb: BufF
    var sinb: BufF
    var qw: BufF
    var lev: BufF
    var wf: BufF
    var gf: BufF
    var mf: BufF
    var vf: BufF
    var cf: BufD
    var gvec: BufF
    var n_mat: Int
    var n_all: Int

    def __init__(out self, ctx: DeviceContext) raises:
        var names = List[String]()
        var sizes = List[Int]()
        var wds = List[Bool]()
        names.append("fq"); sizes.append(D); wds.append(False)
        names.append("e1"); sizes.append(4 * D * 32); wds.append(True)
        names.append("e1b"); sizes.append(4 * D); wds.append(False)
        names.append("e2"); sizes.append(D * 4 * D); wds.append(True)
        names.append("e2b"); sizes.append(D); wds.append(False)
        names.append("er"); sizes.append(D * 32); wds.append(True)
        names.append("erb"); sizes.append(D); wds.append(False)
        names.append("on"); sizes.append(D); wds.append(False)
        names.append("h1"); sizes.append(4 * D * D); wds.append(True)
        names.append("h2"); sizes.append(NHEAD * 4 * D); wds.append(True)
        names.append("hr"); sizes.append(NHEAD * D); wds.append(True)
        for l in range(NL):
            var p = "l" + String(l) + "_"
            names.append(p + "n0"); sizes.append(D); wds.append(False)
            names.append(p + "qkvg"); sizes.append(4 * D * D); wds.append(True)
            names.append(p + "o0"); sizes.append(D * D); wds.append(True)
            names.append(p + "nm"); sizes.append(D); wds.append(False)
            names.append(p + "w12"); sizes.append(2 * FF * D); wds.append(True)
            names.append(p + "w3"); sizes.append(D * FF); wds.append(True)
            if is_var(l):
                names.append(p + "n1"); sizes.append(D); wds.append(False)
                names.append(p + "vg"); sizes.append(2 * D * D); wds.append(True)
                names.append(p + "o1"); sizes.append(D * D); wds.append(True)
        var n_mat = 0
        var n_vec = 0
        for i in range(len(names)):
            if wds[i]:
                n_mat += sizes[i]
            else:
                n_vec += sizes[i]
        self.n_mat = n_mat
        self.n_all = n_mat + n_vec
        self.wf = zeros_f(ctx, n_mat + n_vec)
        self.gf = zeros_f(ctx, n_mat + n_vec)
        self.mf = zeros_f(ctx, n_mat + n_vec)
        self.vf = zeros_f(ctx, n_mat + n_vec)
        self.cf = zeros_d(ctx, n_mat + n_vec)
        self.gvec = self.gf.create_sub_buffer[F32](n_mat, n_vec)
        self.P = List[Param]()
        var mo = 0
        var vo = n_mat
        for i in range(len(names)):
            var off = 0
            if wds[i]:
                off = mo
                mo += sizes[i]
            else:
                off = vo
                vo += sizes[i]
            self.P.append(Param(ctx, names[i], sizes[i], wds[i], off, self.wf, self.gf, self.mf, self.vf, self.cf))
        self.cosb = load_f32(ctx, "mojo/data/cos.f32", 33 * HD)
        self.sinb = load_f32(ctx, "mojo/data/sin.f32", 33 * HD)
        self.qw = load_f32(ctx, "mojo/data/qw.f32", NQ)
        var lv = ctx.enqueue_create_host_buffer[F32](NQ)
        lv[0] = 0.01; lv[1] = 0.05; lv[2] = 0.1; lv[3] = 0.15; lv[4] = 0.2; lv[5] = 0.25; lv[6] = 0.3
        lv[7] = 0.35; lv[8] = 0.4; lv[9] = 0.45; lv[10] = 0.5; lv[11] = 0.55; lv[12] = 0.6; lv[13] = 0.65
        lv[14] = 0.7; lv[15] = 0.75; lv[16] = 0.8; lv[17] = 0.85; lv[18] = 0.9; lv[19] = 0.95; lv[20] = 0.99
        self.lev = ctx.enqueue_create_buffer[F32](NQ)
        ctx.enqueue_copy(self.lev, lv)
        ctx.synchronize()
        self.refresh(ctx)

    def refresh(self, ctx: DeviceContext) raises:
        k_cast_f2d(ctx, self.wf, self.cf, self.n_all)


# ---------------------------------------------------------------- activations
struct LAct(Movable):
    var variate: Bool
    var x1: BufF       # after the time-attention residual
    var x2: BufF       # after the variate residual (unused for time-only layers)
    var h: BufD        # rmsnorm(x) * n0
    var qkvg: BufD
    var opre: BufD     # attention output before the gate
    var o: BufD        # gated attention output
    var h1: BufD
    var vg: BufD
    var o1: BufD
    var hm: BufD
    var w12o: BufD
    var act: BufD      # swiglu output (input of the w3 GEMM)

    def __init__(out self, ctx: DeviceContext, M: Int, variate: Bool) raises:
        self.variate = variate
        self.x1 = ctx.enqueue_create_buffer[F32](M * D)
        self.x2 = ctx.enqueue_create_buffer[F32](M * D if variate else 1)
        self.h = ctx.enqueue_create_buffer[DT](M * D)
        self.qkvg = ctx.enqueue_create_buffer[DT](M * 4 * D)
        self.opre = ctx.enqueue_create_buffer[DT](M * D)
        self.o = ctx.enqueue_create_buffer[DT](M * D)
        self.h1 = ctx.enqueue_create_buffer[DT](M * D if variate else 1)
        self.vg = ctx.enqueue_create_buffer[DT](M * 2 * D if variate else 1)
        self.o1 = ctx.enqueue_create_buffer[DT](M * D if variate else 1)
        self.hm = ctx.enqueue_create_buffer[DT](M * D)
        self.w12o = ctx.enqueue_create_buffer[DT](M * 2 * FF)
        self.act = ctx.enqueue_create_buffer[DT](M * FF)


struct Acts(Movable):
    var R: Int
    var M: Int
    var xin: BufF
    var yb: BufF
    var vb_: BufF
    var z: BufD
    var loc: BufF
    var sc: BufF
    var pre1: BufD
    var e1h: BufD
    var e2o: BufD
    var ero: BufD
    var xs: List[BufF]       # residual input of each layer, xs[NL] = final output
    var la: List[LAct]
    var big: BufD
    var t: BufD
    var u: BufD
    var t2: BufD
    var dx: BufF
    var d: BufD
    var z1: BufD
    var hh: BufD
    var r2: BufD
    var rr: BufD
    var draw: BufD
    var lossb: BufF
    var medb: BufF
    var dd: BufD
    var dz1: BufD
    var dhh: BufD
    var x16: BufF
    var dx16: BufF
    var part: BufF           # per-block partial column sums, NG x 4D max
    var clip: BufF
    var sq: BufF

    def __init__(out self, ctx: DeviceContext, R: Int) raises:
        var M = R * S
        self.R = R
        self.M = M
        self.xin = ctx.enqueue_create_buffer[F32](R * CTX)
        self.yb = ctx.enqueue_create_buffer[F32](R * 32)
        self.vb_ = ctx.enqueue_create_buffer[F32](R * 32)
        self.z = ctx.enqueue_create_buffer[DT](R * L * 32)
        self.loc = ctx.enqueue_create_buffer[F32](R * L)
        self.sc = ctx.enqueue_create_buffer[F32](R * L)
        self.pre1 = ctx.enqueue_create_buffer[DT](R * L * 4 * D)
        self.e1h = ctx.enqueue_create_buffer[DT](R * L * 4 * D)
        self.e2o = ctx.enqueue_create_buffer[DT](R * L * D)
        self.ero = ctx.enqueue_create_buffer[DT](R * L * D)
        self.xs = List[BufF]()
        for _ in range(NL + 1):
            self.xs.append(ctx.enqueue_create_buffer[F32](M * D))
        self.la = List[LAct]()
        for l in range(NL):
            self.la.append(LAct(ctx, M, is_var(l)))
        self.big = ctx.enqueue_create_buffer[DT](M * 2 * FF)
        self.t = ctx.enqueue_create_buffer[DT](M * D)
        self.u = ctx.enqueue_create_buffer[DT](M * FF)
        self.t2 = ctx.enqueue_create_buffer[DT](M * D)
        self.dx = ctx.enqueue_create_buffer[F32](M * D)
        self.d = ctx.enqueue_create_buffer[DT](R * D)
        self.z1 = ctx.enqueue_create_buffer[DT](R * 4 * D)
        self.hh = ctx.enqueue_create_buffer[DT](R * 4 * D)
        self.r2 = ctx.enqueue_create_buffer[DT](R * NHEAD)
        self.rr = ctx.enqueue_create_buffer[DT](R * NHEAD)
        self.draw = ctx.enqueue_create_buffer[DT](R * NHEAD)
        self.lossb = ctx.enqueue_create_buffer[F32](R)
        self.medb = ctx.enqueue_create_buffer[F32](R)
        self.dd = ctx.enqueue_create_buffer[DT](R * D)
        self.dz1 = ctx.enqueue_create_buffer[DT](R * 4 * D)
        self.dhh = ctx.enqueue_create_buffer[DT](R * 4 * D)
        self.x16 = ctx.enqueue_create_buffer[F32](R * D)
        self.dx16 = ctx.enqueue_create_buffer[F32](R * D)
        self.part = ctx.enqueue_create_buffer[F32](NG * 4 * D)
        self.clip = ctx.enqueue_create_buffer[F32](4)
        self.sq = ctx.enqueue_create_buffer[F32](512)


# ---------------------------------------------------------------- forward kernels
def k_prep(ctx: DeviceContext, a: Acts) raises:
    """Causal fp64 Welford stats; per-patch loc/scale and asinh-normalised patches."""
    var R = a.R
    var px = pf(a.xin)
    var pz = pd(a.z)
    var pl = pf(a.loc)
    var ps = pf(a.sc)
    var R64 = Int64(R)

    def k() {var px, var pz, var pl, var ps, var R64}:
        var r = Int(block_idx.x * block_dim.x + thread_idx.x)
        if r >= Int(R64):
            return
        var row = px + r * CTX
        var cumx: Float64 = 0
        var mean: Float64 = 0
        var m2: Float64 = 0
        for t in range(CTX):
            var x = Float64(row[t])
            var prev = mean
            cumx += x
            mean = cumx / Float64(t + 1)
            m2 += (x - prev) * (x - mean)
            if t % 32 == 31:
                var den = Float64(t)
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
                    pz[(r * L + p) * 32 + j] = Scalar[DT](asinh((row[p * 32 + j] - lf) / dn))

    ctx.enqueue_function(k, grid_dim=ceildiv(R, 64), block_dim=64)


def k_silu_bias_out(ctx: DeviceContext, src: BufD, bias: BufF, dst: BufD, rows: Int) raises:
    var p = pd(src)
    var q = pd(dst)
    var pb = pf(bias)
    var tot64 = Int64(rows * 4 * D)

    def k() {var p, var q, var pb, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            var v = Float32(p[i]) + pb[i % (4 * D)]
            q[i] = Scalar[DT](v / (1.0 + exp(-v)))

    ctx.enqueue_function(k, grid_dim=ceildiv(rows * 4 * D, 256), block_dim=256)


def k_embed(ctx: DeviceContext, a: Acts, net: Net, xout: BufF, hout: BufD, n0: BufF) raises:
    """x[b, p] = e2 + b2 + er + erb (p < L), x[b, L] = forecast query; h = rmsnorm(x) * n0."""
    var R = a.R
    var px = pf(xout)
    var ph = pd(hout)
    var pe = pd(a.e2o)
    var pr = pd(a.ero)
    var pb2 = pf(net.P[P_E2B].w)
    var pbr = pf(net.P[P_ERB].w)
    var pfq = pf(net.P[P_FQ].w)
    var pw = pf(n0)

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


def k_add_rms_out(ctx: DeviceContext, xin: BufF, tt: BufD, w: BufF, xout: BufF, hout: BufD, M: Int, write_h: Bool) raises:
    """xout = xin + t (fp32 residual); hout = rmsnorm(xout) * w."""
    var px = pf(xin)
    var pt = pd(tt)
    var pw = pf(w)
    var po = pf(xout)
    var ph = pd(hout)
    var wh = write_h

    def k() {var px, var pt, var pw, var po, var ph, var wh}:
        var r = Int(block_idx.x)
        var t = Int(thread_idx.x)
        var v0 = px[r * D + t] + Float32(pt[r * D + t])
        var v1 = px[r * D + t + 256] + Float32(pt[r * D + t + 256])
        var v2 = px[r * D + t + 512] + Float32(pt[r * D + t + 512])
        po[r * D + t] = v0
        po[r * D + t + 256] = v1
        po[r * D + t + 512] = v2
        if wh:
            var tot = block_sum[block_size=256](v0 * v0 + v1 * v1 + v2 * v2)
            var inv = Float32(1.0) / sqrt(tot / Float32(D) + 1e-6)
            ph[r * D + t] = Scalar[DT](v0 * inv * pw[t])
            ph[r * D + t + 256] = Scalar[DT](v1 * inv * pw[t + 256])
            ph[r * D + t + 512] = Scalar[DT](v2 * inv * pw[t + 512])

    ctx.enqueue_function(k, grid_dim=M, block_dim=256)


def k_attn_train(ctx: DeviceContext, net: Net, la: LAct, R: Int) raises:
    """Causal attention with RoPE + sigmoid gate over the S tokens of each (row, head). Writes opre (ungated) and o (gated)."""
    var pq = pd(la.qkvg)
    var po = pd(la.o)
    var pp = pd(la.opre)
    var pc = pf(net.cosb)
    var ps = pf(net.sinb)
    comptime NT = S * 8
    comptime KS = 65

    def k() {var pq, var po, var pp, var pc, var ps}:
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
        var res0 = SIMD[DT, 8](0)
        comptime for i in range(8):
            var g = Float32(gv[i])
            res0[i] = Scalar[DT](acc[i])
            res[i] = Scalar[DT](Float32(res0[i]) / (Float32(1.0) + exp(-g)))
        (po + (b * S + qi) * D + h * HD + 8 * v8).unsafe_store[width=8, alignment=16](res)
        (pp + (b * S + qi) * D + h * HD + 8 * v8).unsafe_store[width=8, alignment=16](res0)

    ctx.enqueue_function(k, grid_dim=R * NH, block_dim=NT)


def k_vgate_fwd(ctx: DeviceContext, vg: BufD, o1: BufD, M: Int) raises:
    var pb = pd(vg)
    var po = pd(o1)
    var tot64 = Int64(M * D)

    def k() {var pb, var po, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            var r = i // D
            var c = i % D
            var v = Float32(pb[r * 2 * D + c])
            var g = Float32(pb[r * 2 * D + D + c])
            po[i] = Scalar[DT](v / (Float32(1.0) + exp(-g)))

    ctx.enqueue_function(k, grid_dim=ceildiv(M * D, 256), block_dim=256)


def k_swiglu(ctx: DeviceContext, w12o: BufD, uu: BufD, M: Int) raises:
    var pb = pd(w12o)
    var pu = pd(uu)
    var tot64 = Int64(M * FF)

    def k() {var pb, var pu, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            var r = i // FF
            var c = i % FF
            var g = Float32(pb[r * 2 * FF + c])
            var u = Float32(pb[r * 2 * FF + FF + c])
            pu[i] = Scalar[DT](g * (u / (Float32(1.0) + exp(-u))))

    ctx.enqueue_function(k, grid_dim=ceildiv(M * FF, 256), block_dim=256)


def k_final_norm(ctx: DeviceContext, a: Acts, on: BufF) raises:
    """d[b] = rmsnorm(xs[NL][b, L]) * out_norm; also copies the fp32 token row to x16."""
    var R = a.R
    var px = pf(a.xs[NL])
    var pdd = pd(a.d)
    var p16 = pf(a.x16)
    var pw = pf(on)

    def k() {var px, var pdd, var p16, var pw}:
        var b = Int(block_idx.x)
        var t = Int(thread_idx.x)
        var row = px + (b * S + (S - 1)) * D
        var v0 = row[t]
        var v1 = row[t + 256]
        var v2 = row[t + 512]
        p16[b * D + t] = v0
        p16[b * D + t + 256] = v1
        p16[b * D + t + 512] = v2
        var tot = block_sum[block_size=256](v0 * v0 + v1 * v1 + v2 * v2)
        var inv = Float32(1.0) / sqrt(tot / Float32(D) + 1e-6)
        pdd[b * D + t] = Scalar[DT](v0 * inv * pw[t])
        pdd[b * D + t + 256] = Scalar[DT](v1 * inv * pw[t + 256])
        pdd[b * D + t + 512] = Scalar[DT](v2 * inv * pw[t + 512])

    ctx.enqueue_function(k, grid_dim=R, block_dim=256)


def k_silu_copy(ctx: DeviceContext, src: BufD, dst: BufD, n: Int) raises:
    var ps = pd(src)
    var po = pd(dst)
    var n64 = Int64(n)

    def k() {var ps, var po, var n64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(n64):
            var v = Float32(ps[i])
            po[i] = Scalar[DT](v / (Float32(1.0) + exp(-v)))

    ctx.enqueue_function(k, grid_dim=ceildiv(n, 256), block_dim=256)


def k_loss(ctx: DeviceContext, a: Acts, net: Net, grad_scale: Float32) raises:
    """Head post-processing + pinball loss + dloss/draw. One thread per row.
    draw[b, k, i] = grad_scale * w_k * valid * dpinball/draw  (grad_scale = loss_scale / (sum(w*valid) * NQ * LOSS_SCALE))."""
    var R = a.R
    var pr2 = pd(a.r2)
    var prr = pd(a.rr)
    var pl = pf(a.loc)
    var ps = pf(a.sc)
    var py = pf(a.yb)
    var pv = pf(a.vb_)
    var pqw = pf(net.qw)
    var plv = pf(net.lev)
    var pdr = pd(a.draw)
    var plo = pf(a.lossb)
    var pmed = pf(a.medb)
    var R64 = Int64(R)

    def k() {var pr2, var prr, var pl, var ps, var py, var pv, var pqw, var plv, var pdr, var plo, var pmed, var R64, var grad_scale}:
        var b = Int(block_idx.x * block_dim.x + thread_idx.x)
        if b >= Int(R64):
            return
        var lf = pl[b * L + L - 1]
        var sf = ps[b * L + L - 1]
        var tot: Float32 = 0
        for kk in range(32):
            var wk: Float32 = 0.5 / 31.0
            if kk == 0:
                wk = 1.0
            var vk = pv[b * 32 + kk]
            var raw = Array[Float32, NQ](fill=0)
            var sg = Array[Float32, NQ](fill=0)
            var inc = Array[Float32, NQ](fill=0)
            for q in range(NQ):
                var r = Float32(pr2[b * NHEAD + kk * NQ + q]) + Float32(prr[b * NHEAD + kk * NQ + q])
                raw[q] = r
                var s = Float32(1.0) / (Float32(1.0) + exp(-r))
                sg[q] = s
                var sp = r
                if r <= 20.0:
                    sp = log(Float32(1.0) + exp(r))
                inc[q] = pqw[q] * sp
            var qn = Array[Float32, NQ](fill=0)
            var med = raw[10]
            qn[10] = med
            var acc: Float32 = 0
            for i in range(11, NQ):
                acc += inc[i]
                qn[i] = med + acc
            acc = 0
            for i in range(9, -1, -1):
                acc += inc[i]
                qn[i] = med - acc
            var dqn = Array[Float32, NQ](fill=0)
            var yk = py[b * 32 + kk]
            var pin: Float32 = 0
            for q in range(NQ):
                var qq = sinh(qn[q]) * sf + lf
                if kk == 0 and q == 10:
                    pmed[b] = qq
                var e = yk - qq
                var tau = plv[q]
                var dq: Float32
                if e > 0:
                    pin += tau * e
                    dq = -tau
                else:
                    pin += (tau - 1.0) * e
                    dq = 1.0 - tau
                dqn[q] = dq * cosh(qn[q]) * sf
            tot += wk * vk * pin / Float32(NQ) / 0.02
            var gs = grad_scale * wk * vk
            # back through the monotone construction
            var dmed: Float32 = 0
            for q in range(NQ):
                dmed += dqn[q]
            var dr = Array[Float32, NQ](fill=0)
            var suffix: Float32 = 0
            for i in range(NQ - 1, 10, -1):
                suffix += dqn[i]
                dr[i] = suffix * pqw[i] * sg[i]
            var prefix: Float32 = 0
            for i in range(0, 10):
                prefix += dqn[i]
                dr[i] = -prefix * pqw[i] * sg[i]
            dr[10] = dmed
            for q in range(NQ):
                pdr[b * NHEAD + kk * NQ + q] = Scalar[DT](dr[q] * gs)
        plo[b] = tot

    ctx.enqueue_function(k, grid_dim=ceildiv(R, 64), block_dim=64)



struct TP(Movable):
    var on: Bool
    var acc: List[Float64]
    var last: Int

    def __init__(out self, on: Bool):
        self.on = on
        self.acc = List[Float64]()
        for _ in range(16):
            self.acc.append(0.0)
        self.last = Int(perf_counter_ns())

    def start(mut self, ctx: DeviceContext) raises:
        if self.on:
            ctx.synchronize()
            self.last = Int(perf_counter_ns())

    def lap(mut self, ctx: DeviceContext, i: Int) raises:
        if self.on:
            ctx.synchronize()
            var n = Int(perf_counter_ns())
            self.acc[i] += Float64(n - self.last)
            self.last = n


def tp_name(i: Int) -> String:
    if i == 0: return "gather+zero"
    if i == 1: return "forward"
    if i == 2: return "loss"
    if i == 3: return "bwd head"
    if i == 4: return "bwd mlp"
    if i == 5: return "bwd variate"
    if i == 6: return "bwd attn"
    if i == 7: return "bwd embed"
    if i == 8: return "clip+adam"
    return "other"


def forward_train(ctx: DeviceContext, net: Net, a: Acts) raises:
    var R = a.R
    var M = R * S
    k_prep(ctx, a)
    gnt[4 * D, 32](ctx, pd(a.pre1), pd(a.z), pd(net.P[P_E1].c), R * L)
    k_silu_bias_out(ctx, a.pre1, net.P[P_E1B].w, a.e1h, R * L)
    gnt[D, 4 * D](ctx, pd(a.e2o), pd(a.e1h), pd(net.P[P_E2].c), R * L)
    gnt[D, 32](ctx, pd(a.ero), pd(a.z), pd(net.P[P_ER].c), R * L)
    k_embed(ctx, a, net, a.xs[0], a.la[0].h, net.P[lbase(0) + O_N0].w)
    for l in range(NL):
        var b0 = lbase(l)
        ref ac = a.la[l]
        gnt[4 * D, D](ctx, pd(ac.qkvg), pd(ac.h), pd(net.P[b0 + O_QKVG].c), M)
        k_attn_train(ctx, net, ac, R)
        gnt[D, D](ctx, pd(a.t), pd(ac.o), pd(net.P[b0 + O_O0].c), M)
        if ac.variate:
            k_add_rms_out(ctx, a.xs[l], a.t, net.P[b0 + O_N1].w, ac.x1, ac.h1, M, True)
            gnt[2 * D, D](ctx, pd(ac.vg), pd(ac.h1), pd(net.P[b0 + O_VG].c), M)
            k_vgate_fwd(ctx, ac.vg, ac.o1, M)
            gnt[D, D](ctx, pd(a.t), pd(ac.o1), pd(net.P[b0 + O_O1].c), M)
            k_add_rms_out(ctx, ac.x1, a.t, net.P[b0 + O_NM].w, ac.x2, ac.hm, M, True)
        else:
            k_add_rms_out(ctx, a.xs[l], a.t, net.P[b0 + O_NM].w, ac.x1, ac.hm, M, True)
        gnt[2 * FF, D](ctx, pd(ac.w12o), pd(ac.hm), pd(net.P[b0 + O_W12].c), M)
        k_swiglu(ctx, ac.w12o, ac.act, M)
        gnt[D, FF](ctx, pd(a.t), pd(ac.act), pd(net.P[b0 + O_W3].c), M)
        if l + 1 < NL:
            if ac.variate:
                k_add_rms_out(ctx, ac.x2, a.t, net.P[lbase(l + 1) + O_N0].w, a.xs[l + 1], a.la[l + 1].h, M, True)
            else:
                k_add_rms_out(ctx, ac.x1, a.t, net.P[lbase(l + 1) + O_N0].w, a.xs[l + 1], a.la[l + 1].h, M, True)
        else:
            if ac.variate:
                k_add_rms_out(ctx, ac.x2, a.t, net.P[P_ON].w, a.xs[NL], a.t2, M, False)
            else:
                k_add_rms_out(ctx, ac.x1, a.t, net.P[P_ON].w, a.xs[NL], a.t2, M, False)
    k_final_norm(ctx, a, net.P[P_ON].w)
    gnt[4 * D, D](ctx, pd(a.z1), pd(a.d), pd(net.P[P_H1].c), R)
    k_silu_copy(ctx, a.z1, a.hh, R * 4 * D)
    gnt[NHEAD, 4 * D](ctx, pd(a.r2), pd(a.hh), pd(net.P[P_H2].c), R)
    gnt[NHEAD, D](ctx, pd(a.rr), pd(a.d), pd(net.P[P_HR].c), R)



# ---------------------------------------------------------------- backward kernels
def k_f2d(ctx: DeviceContext, src: BufF, dst: BufD, n: Int) raises:
    k_cast_f2d(ctx, src, dst, n)


def k_rms_bwd(ctx: DeviceContext, x: PF, dh: PD, w: PF, dxres: PF, dxout: PF, dxd: PD, part: PF, rows: Int, use_res: Bool, write_d: Bool) raises:
    """RMSNorm backward. dxout = dxres + r*w*dh - x*r^3/D*sum(dh*w*x); per-block partial dw = sum_rows dh*x*r (NG x D)."""
    var rows64 = Int64(rows)
    var ur = use_res
    var wdd = write_d

    def k() {var x, var dh, var w, var dxres, var dxout, var dxd, var part, var rows64, var ur, var wdd}:
        var bk = Int(block_idx.x)
        var t = Int(thread_idx.x)
        var a0: Float32 = 0
        var a1: Float32 = 0
        var a2: Float32 = 0
        var w0 = w[t]
        var w1 = w[t + 256]
        var w2 = w[t + 512]
        for r in range(bk, Int(rows64), NG):
            var x0 = x[r * D + t]
            var x1 = x[r * D + t + 256]
            var x2 = x[r * D + t + 512]
            var d0 = Float32(dh[r * D + t])
            var d1 = Float32(dh[r * D + t + 256])
            var d2 = Float32(dh[r * D + t + 512])
            var ss = block_sum[block_size=256](x0 * x0 + x1 * x1 + x2 * x2)
            var inv = Float32(1.0) / sqrt(ss / Float32(D) + 1e-6)
            var dot = block_sum[block_size=256](d0 * w0 * x0 + d1 * w1 * x1 + d2 * w2 * x2)
            var c = inv * inv * inv / Float32(D) * dot
            var r0 = inv * w0 * d0 - x0 * c
            var r1 = inv * w1 * d1 - x1 * c
            var r2 = inv * w2 * d2 - x2 * c
            if ur:
                r0 += dxres[r * D + t]
                r1 += dxres[r * D + t + 256]
                r2 += dxres[r * D + t + 512]
            dxout[r * D + t] = r0
            dxout[r * D + t + 256] = r1
            dxout[r * D + t + 512] = r2
            if wdd:
                dxd[r * D + t] = Scalar[DT](r0)
                dxd[r * D + t + 256] = Scalar[DT](r1)
                dxd[r * D + t + 512] = Scalar[DT](r2)
            a0 += d0 * x0 * inv
            a1 += d1 * x1 * inv
            a2 += d2 * x2 * inv
        part[bk * D + t] = a0
        part[bk * D + t + 256] = a1
        part[bk * D + t + 512] = a2

    ctx.enqueue_function(k, grid_dim=NG, block_dim=256)


def k_dw_reduce(ctx: DeviceContext, part: PF, g: PF) raises:
    def k() {var part, var g}:
        var j = Int(block_idx.x * block_dim.x + thread_idx.x)
        if j < D:
            var s: Float32 = 0
            for b in range(NG):
                s += part[b * D + j]
            g[j] += s

    ctx.enqueue_function(k, grid_dim=ceildiv(D, 256), block_dim=256)


def rms_bwd(ctx: DeviceContext, a: Acts, x: BufF, dh: BufD, wp: Param, dxres: BufF, dxout: BufF, rows: Int, use_res: Bool) raises:
    k_rms_bwd(ctx, pf(x), pd(dh), pf(wp.w), pf(dxres), pf(dxout), pd(a.t), pf(a.part), rows, use_res, True)
    k_dw_reduce(ctx, pf(a.part), pf(wp.g))


def k_swiglu_bwd(ctx: DeviceContext, w12o: BufD, da: BufD, dw12o: BufD, M: Int) raises:
    var pb = pd(w12o)
    var pa = pd(da)
    var po = pd(dw12o)
    var tot64 = Int64(M * FF)

    def k() {var pb, var pa, var po, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            var r = i // FF
            var c = i % FF
            var g = Float32(pb[r * 2 * FF + c])
            var u = Float32(pb[r * 2 * FF + FF + c])
            var dav = Float32(pa[i])
            var s = Float32(1.0) / (Float32(1.0) + exp(-u))
            po[r * 2 * FF + c] = Scalar[DT](dav * u * s)
            po[r * 2 * FF + FF + c] = Scalar[DT](dav * g * s * (Float32(1.0) + u * (Float32(1.0) - s)))

    ctx.enqueue_function(k, grid_dim=ceildiv(M * FF, 256), block_dim=256)


def k_vgate_bwd(ctx: DeviceContext, vg: BufD, do1: BufD, dvg: BufD, M: Int) raises:
    var pb = pd(vg)
    var pa = pd(do1)
    var po = pd(dvg)
    var tot64 = Int64(M * D)

    def k() {var pb, var pa, var po, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            var r = i // D
            var c = i % D
            var v = Float32(pb[r * 2 * D + c])
            var g = Float32(pb[r * 2 * D + D + c])
            var dov = Float32(pa[i])
            var s = Float32(1.0) / (Float32(1.0) + exp(-g))
            po[r * 2 * D + c] = Scalar[DT](dov * s)
            po[r * 2 * D + D + c] = Scalar[DT](dov * v * s * (Float32(1.0) - s))

    ctx.enqueue_function(k, grid_dim=ceildiv(M * D, 256), block_dim=256)


def k_silu_bwd(ctx: DeviceContext, pre: BufD, dh: BufD, dout: BufD, bias: BufF, has_bias: Bool, n: Int, ncols: Int) raises:
    """dout = dh * silu'(pre [+ bias])."""
    var pp = pd(pre)
    var pg = pd(dh)
    var po = pd(dout)
    var pb = pf(bias)
    var hb = has_bias
    var n64 = Int64(n)
    var nc = Int64(ncols)

    def k() {var pp, var pg, var po, var pb, var hb, var n64, var nc}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(n64):
            var v = Float32(pp[i])
            if hb:
                v += pb[i % Int(nc)]
            var s = Float32(1.0) / (Float32(1.0) + exp(-v))
            po[i] = Scalar[DT](Float32(pg[i]) * s * (Float32(1.0) + v * (Float32(1.0) - s)))

    ctx.enqueue_function(k, grid_dim=ceildiv(n, 256), block_dim=256)


def k_attn_bwd(ctx: DeviceContext, net: Net, la: LAct, dao: BufD, dqkvg: BufD, R: Int) raises:
    """Attention + gate + RoPE backward. dao = grad of the gated attention output (M, D); writes dqkvg = [dq|dk|dv|dgate]."""
    var pq = pd(la.qkvg)
    var pp = pd(la.opre)
    var pdo = pd(dao)
    var pdq = pd(dqkvg)
    var pc = pf(net.cosb)
    var ps = pf(net.sinb)
    comptime NT = S * 8
    comptime KS = 65

    def k() {var pq, var pp, var pdo, var pdq, var pc, var ps}:
        var h = Int(block_idx.x) % NH
        var b = Int(block_idx.x) // NH
        var t = Int(thread_idx.x)
        var qs = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var ks = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var vs = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var dos = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var dqs = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var dks = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var P = stack_allocation[S * S, Float32, address_space=AddressSpace.SHARED]()
        var dS = stack_allocation[S * S, Float32, address_space=AddressSpace.SHARED]()
        var rds = stack_allocation[64, Float32, address_space=AddressSpace.SHARED]()
        var j8 = t // 8
        var v8 = t % 8
        var rowp = pq + (b * S + j8) * (4 * D) + h * HD
        var vv = (rowp + 2 * D + 8 * v8).unsafe_load[width=8, alignment=16]()
        var gv = (rowp + 3 * D + 8 * v8).unsafe_load[width=8, alignment=16]()
        var dov = (pdo + (b * S + j8) * D + h * HD + 8 * v8).unsafe_load[width=8, alignment=16]()
        var opv = (pp + (b * S + j8) * D + h * HD + 8 * v8).unsafe_load[width=8, alignment=16]()
        var dgv = SIMD[DT, 8](0)
        comptime for i in range(8):
            var sg = Float32(1.0) / (Float32(1.0) + exp(-Float32(gv[i])))
            vs[j8 * KS + 8 * v8 + i] = Float32(vv[i])
            dos[j8 * KS + 8 * v8 + i] = Float32(dov[i]) * sg
            dgv[i] = Scalar[DT](Float32(dov[i]) * Float32(opv[i]) * sg * (Float32(1.0) - sg))
        (pdq + (b * S + j8) * (4 * D) + 3 * D + h * HD + 8 * v8).unsafe_store[width=8, alignment=16](dgv)
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
                var dp: Float32 = 0
                for d in range(HD):
                    a += qs[qi * KS + d] * ks[kj * KS + d]
                    dp += dos[qi * KS + d] * vs[kj * KS + d]
                P[idx] = a * 0.125
                dS[idx] = dp
            else:
                P[idx] = -3.0e38
                dS[idx] = 0
        barrier()
        if t < S:
            var mx: Float32 = -3.0e38
            for j in range(t + 1):
                mx = max(mx, P[t * S + j])
            var tot: Float32 = 0
            for j in range(t + 1):
                var e = exp(P[t * S + j] - mx)
                P[t * S + j] = e
                tot += e
            var inv = Float32(1.0) / tot
            var rd: Float32 = 0
            for j in range(t + 1):
                var pv = P[t * S + j] * inv
                P[t * S + j] = pv
                rd += pv * dS[t * S + j]
            rds[t] = rd
        barrier()
        for idx in range(t, S * S, NT):
            var qi = idx // S
            var kj = idx % S
            if kj <= qi:
                dS[idx] = P[idx] * (dS[idx] - rds[qi]) * 0.125
        barrier()
        var dvo = SIMD[DT, 8](0)
        comptime for e in range(8):
            var d = 8 * v8 + e
            var dqp: Float32 = 0
            for j in range(j8 + 1):
                dqp += dS[j8 * S + j] * ks[j * KS + d]
            var dkp: Float32 = 0
            var dvp: Float32 = 0
            for ii in range(j8, S):
                dkp += dS[ii * S + j8] * qs[ii * KS + d]
                dvp += P[ii * S + j8] * dos[ii * KS + d]
            dqs[j8 * KS + d] = dqp
            dks[j8 * KS + d] = dkp
            dvo[e] = Scalar[DT](dvp)
        (pdq + (b * S + j8) * (4 * D) + 2 * D + h * HD + 8 * v8).unsafe_store[width=8, alignment=16](dvo)
        barrier()
        if t < S * 4:
            var j = t // 4
            var u = t % 4
            var oq0 = SIMD[DT, 8](0)
            var oq1 = SIMD[DT, 8](0)
            var ok0 = SIMD[DT, 8](0)
            var ok1 = SIMD[DT, 8](0)
            comptime for e in range(8):
                var c = pc[j * HD + 8 * u + e]
                var s = ps[j * HD + 8 * u + e]
                var qa = dqs[j * KS + 8 * u + e]
                var qb = dqs[j * KS + 32 + 8 * u + e]
                var ka = dks[j * KS + 8 * u + e]
                var kb = dks[j * KS + 32 + 8 * u + e]
                oq0[e] = Scalar[DT](qa * c + qb * s)
                oq1[e] = Scalar[DT](qb * c - qa * s)
                ok0[e] = Scalar[DT](ka * c + kb * s)
                ok1[e] = Scalar[DT](kb * c - ka * s)
            var base = pdq + (b * S + j) * (4 * D) + h * HD
            (base + 8 * u).unsafe_store[width=8, alignment=16](oq0)
            (base + 32 + 8 * u).unsafe_store[width=8, alignment=16](oq1)
            (base + D + 8 * u).unsafe_store[width=8, alignment=16](ok0)
            (base + D + 32 + 8 * u).unsafe_store[width=8, alignment=16](ok1)

    ctx.enqueue_function(k, grid_dim=R * NH, block_dim=NT)


def k_colsum_d(ctx: DeviceContext, src: BufD, rows: Int, cols: Int, d1: PF, d2: PF, two: Bool) raises:
    """d1 (and d2) [j] += sum_i src[i, j]."""
    var ps = pd(src)
    var rows64 = Int64(rows)
    var cols64 = Int64(cols)
    var tw = two

    def k() {var ps, var d1, var d2, var rows64, var cols64, var tw}:
        var j = Int(block_idx.x * block_dim.x + thread_idx.x)
        if j < Int(cols64):
            var s: Float32 = 0
            for i in range(Int(rows64)):
                s += Float32(ps[i * Int(cols64) + j])
            d1[j] += s
            if tw:
                d2[j] += s

    ctx.enqueue_function(k, grid_dim=ceildiv(cols, 256), block_dim=256)


def k_fq_grad(ctx: DeviceContext, dx: BufF, dfq: PF, R: Int) raises:
    """dfq[j] += sum_b dx[b*S + L, j]."""
    var px = pf(dx)
    var R64 = Int64(R)

    def k() {var px, var dfq, var R64}:
        var j = Int(block_idx.x * block_dim.x + thread_idx.x)
        if j < D:
            var s: Float32 = 0
            for b in range(Int(R64)):
                s += px[(b * S + L) * D + j]
            dfq[j] += s

    ctx.enqueue_function(k, grid_dim=ceildiv(D, 256), block_dim=256)


def k_embed_de(ctx: DeviceContext, dx: BufF, de: BufD, R: Int) raises:
    """de[b*L + p] = dx[b*S + p] for p < L, cast to DT."""
    var px = pf(dx)
    var po = pd(de)
    var tot64 = Int64(R * L * D)

    def k() {var px, var po, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            var r = i // D
            var c = i % D
            var b = r // L
            var p = r % L
            po[i] = Scalar[DT](px[(b * S + p) * D + c])

    ctx.enqueue_function(k, grid_dim=ceildiv(R * L * D, 256), block_dim=256)


def k_scatter16(ctx: DeviceContext, dx16: BufF, dx: BufF, R: Int) raises:
    """dx = 0 except dx[b*S + L] = dx16[b]."""
    var p16 = pf(dx16)
    var px = pf(dx)
    var tot64 = Int64(R * S * D)

    def k() {var p16, var px, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            var r = i // D
            var c = i % D
            var b = r // S
            var p = r % S
            if p == L:
                px[i] = p16[b * D + c]
            else:
                px[i] = 0

    ctx.enqueue_function(k, grid_dim=ceildiv(R * S * D, 256), block_dim=256)


def zero_grads(ctx: DeviceContext, net: Net) raises:
    ctx.enqueue_memset(net.gvec, 0)     # norm/bias/fq gradients accumulate with +=; matrix gradients are overwritten (beta=0)


def backward_train(ctx: DeviceContext, net: Net, a: Acts, mut tp: TP) raises:
    var R = a.R
    var M = R * S
    tp.start(ctx)
    # ---- head
    gnn[NHEAD, D](ctx, pd(a.dd), pd(a.draw), pd(net.P[P_HR].c), R)
    gtn[NHEAD, D](ctx, pf(net.P[P_HR].g), pd(a.draw), pd(a.d), R)
    gnn[NHEAD, 4 * D](ctx, pd(a.dhh), pd(a.draw), pd(net.P[P_H2].c), R)
    gtn[NHEAD, 4 * D](ctx, pf(net.P[P_H2].g), pd(a.draw), pd(a.hh), R)
    k_silu_bwd(ctx, a.z1, a.dhh, a.dz1, net.P[P_ON].w, False, R * 4 * D, 4 * D)
    gtn[4 * D, D](ctx, pf(net.P[P_H1].g), pd(a.dz1), pd(a.d), R)
    gnn[4 * D, D](ctx, pd(a.dd), pd(a.dz1), pd(net.P[P_H1].c), R, 1.0)
    k_rms_bwd(ctx, pf(a.x16), pd(a.dd), pf(net.P[P_ON].w), pf(a.dx16), pf(a.dx16), pd(a.t), pf(a.part), R, False, False)
    k_dw_reduce(ctx, pf(a.part), pf(net.P[P_ON].g))
    k_scatter16(ctx, a.dx16, a.dx, R)
    k_f2d(ctx, a.dx, a.t, M * D)
    tp.lap(ctx, 3)
    # ---- layers
    for li in range(NL):
        var l = NL - 1 - li
        var b0 = lbase(l)
        ref ac = a.la[l]
        # MLP
        gtn[D, FF](ctx, pf(net.P[b0 + O_W3].g), pd(a.t), pd(ac.act), M)
        gnn[D, FF](ctx, pd(a.u), pd(a.t), pd(net.P[b0 + O_W3].c), M)
        k_swiglu_bwd(ctx, ac.w12o, a.u, a.big, M)
        gtn[2 * FF, D](ctx, pf(net.P[b0 + O_W12].g), pd(a.big), pd(ac.hm), M)
        gnn[2 * FF, D](ctx, pd(a.t2), pd(a.big), pd(net.P[b0 + O_W12].c), M)
        if ac.variate:
            rms_bwd(ctx, a, ac.x2, a.t2, net.P[b0 + O_NM], a.dx, a.dx, M, True)
            tp.lap(ctx, 4)
            gtn[D, D](ctx, pf(net.P[b0 + O_O1].g), pd(a.t), pd(ac.o1), M)
            gnn[D, D](ctx, pd(a.t2), pd(a.t), pd(net.P[b0 + O_O1].c), M)
            k_vgate_bwd(ctx, ac.vg, a.t2, a.big, M)
            gtn[2 * D, D](ctx, pf(net.P[b0 + O_VG].g), pd(a.big), pd(ac.h1), M)
            gnn[2 * D, D](ctx, pd(a.t2), pd(a.big), pd(net.P[b0 + O_VG].c), M)
            rms_bwd(ctx, a, ac.x1, a.t2, net.P[b0 + O_N1], a.dx, a.dx, M, True)
            tp.lap(ctx, 5)
        else:
            rms_bwd(ctx, a, ac.x1, a.t2, net.P[b0 + O_NM], a.dx, a.dx, M, True)
            tp.lap(ctx, 4)
        # time attention
        gtn[D, D](ctx, pf(net.P[b0 + O_O0].g), pd(a.t), pd(ac.o), M)
        gnn[D, D](ctx, pd(a.t2), pd(a.t), pd(net.P[b0 + O_O0].c), M)
        k_attn_bwd(ctx, net, ac, a.t2, a.big, R)
        gtn[4 * D, D](ctx, pf(net.P[b0 + O_QKVG].g), pd(a.big), pd(ac.h), M)
        gnn[4 * D, D](ctx, pd(a.t2), pd(a.big), pd(net.P[b0 + O_QKVG].c), M)
        rms_bwd(ctx, a, a.xs[l], a.t2, net.P[b0 + O_N0], a.dx, a.dx, M, True)
        tp.lap(ctx, 6)
    # ---- embedding
    k_fq_grad(ctx, a.dx, pf(net.P[P_FQ].g), R)
    k_embed_de(ctx, a.dx, a.t, R)
    k_colsum_d(ctx, a.t, R * L, D, pf(net.P[P_E2B].g), pf(net.P[P_ERB].g), True)
    gtn[D, 4 * D](ctx, pf(net.P[P_E2].g), pd(a.t), pd(a.e1h), R * L)
    gtn[D, 32](ctx, pf(net.P[P_ER].g), pd(a.t), pd(a.z), R * L)
    gnn[D, 4 * D](ctx, pd(a.big), pd(a.t), pd(net.P[P_E2].c), R * L)
    k_silu_bwd(ctx, a.pre1, a.big, a.e1h, net.P[P_E1B].w, True, R * L * 4 * D, 4 * D)
    k_colsum_d(ctx, a.e1h, R * L, 4 * D, pf(net.P[P_E1B].g), pf(net.P[P_E1B].g), False)
    gtn[4 * D, 32](ctx, pf(net.P[P_E1].g), pd(a.e1h), pd(a.z), R * L)
    tp.lap(ctx, 7)


def host_f32(ctx: DeviceContext, b: BufF, n: Int) raises -> List[Float32]:
    var hb = ctx.enqueue_create_host_buffer[F32](n)
    ctx.enqueue_copy(hb, b)
    ctx.synchronize()
    var out = List[Float32](length=n, fill=0.0)
    for i in range(n):
        out[i] = hb[i]
    return out^



# ---------------------------------------------------------------- optimizer
def k_sumsq(ctx: DeviceContext, net: Net, part: PF) raises:
    """part[blk] = sum of squares of the flat gradient (grid-stride, 4-wide)."""
    var pg = pf(net.gf)
    var n4 = Int64(net.n_all // 4)
    comptime NB = 512

    def k() {var pg, var part, var n4}:
        var s: Float32 = 0
        var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
        while i < Int(n4):
            var v = (pg + 4 * i).unsafe_load[width=4, alignment=16]()
            s += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
            i += NB * 256
        var tot = block_sum[block_size=256](s)
        if Int(thread_idx.x) == 0:
            part[Int(block_idx.x)] = tot

    ctx.enqueue_function(k, grid_dim=NB, block_dim=256)


def k_clip(ctx: DeviceContext, a: Acts, maxnorm: Float32, ls: Float32) raises:
    """clip[0] = unscale * min(1, maxnorm / ||g/ls||)  (applied to the raw gradients); clip[1] = ||g/ls||."""
    var part = pf(a.sq)
    var pc = pf(a.clip)

    def k() {var part, var pc, var maxnorm, var ls}:
        var s: Float32 = 0
        var i = Int(thread_idx.x)
        while i < 512:
            s += part[i]
            i += 256
        var tot = block_sum[block_size=256](s)
        if Int(thread_idx.x) == 0:
            var gn = sqrt(tot) / ls
            var f: Float32 = 1.0
            if gn > maxnorm:
                f = maxnorm / (gn + 1e-6)
            pc[0] = f / ls
            pc[1] = gn

    ctx.enqueue_function(k, grid_dim=1, block_dim=256)


def k_adam(ctx: DeviceContext, net: Net, clip: PF, lr: Float32, b1: Float32, b2: Float32, eps: Float32, wd: Float32, bc1: Float32, bc2: Float32) raises:
    """AdamW over the whole flat parameter vector (matrices first -> decayed, then norms/biases -> not decayed)."""
    var pw = pf(net.wf)
    var pg = pf(net.gf)
    var pm = pf(net.mf)
    var pv = pf(net.vf)
    var pc = pd(net.cf)
    var n4 = Int64(net.n_all // 4)
    var nm4 = Int64(net.n_mat // 4)

    def k() {var pw, var pg, var pm, var pv, var pc, var clip, var n4, var nm4, var lr, var b1, var b2, var eps, var wd, var bc1, var bc2}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(n4):
            var cs = clip[0]
            var g = (pg + 4 * i).unsafe_load[width=4, alignment=16]() * cs
            var m0 = (pm + 4 * i).unsafe_load[width=4, alignment=16]()
            var v0 = (pv + 4 * i).unsafe_load[width=4, alignment=16]()
            var w0 = (pw + 4 * i).unsafe_load[width=4, alignment=16]()
            var m = b1 * m0 + (1.0 - b1) * g
            var v = b2 * v0 + (1.0 - b2) * g * g
            var decay = lr * wd if i < Int(nm4) else Float32(0.0)
            var w = w0 * (1.0 - decay)
            comptime for e in range(4):
                w[e] = w[e] - lr * (m[e] / bc1) / (sqrt(v[e] / bc2) + eps)
            (pm + 4 * i).unsafe_store[width=4, alignment=16](m)
            (pv + 4 * i).unsafe_store[width=4, alignment=16](v)
            (pw + 4 * i).unsafe_store[width=4, alignment=16](w)
            var c4 = SIMD[DT, 4](0)
            comptime for e in range(4):
                c4[e] = Scalar[DT](w[e])
            (pc + 4 * i).unsafe_store[width=4, alignment=8](c4)

    ctx.enqueue_function(k, grid_dim=ceildiv(net.n_all // 4, 256), block_dim=256)


def optimizer_step(ctx: DeviceContext, net: Net, a: Acts, step: Int, lr: Float32, ls: Float32, maxnorm: Float32) raises:
    k_sumsq(ctx, net, pf(a.sq))
    k_clip(ctx, a, maxnorm, ls)
    var bc1 = Float32(1.0) - Float32(pow_f(0.9, step))
    var bc2 = Float32(1.0) - Float32(pow_f(0.95, step))
    k_adam(ctx, net, pf(a.clip), lr, 0.9, 0.95, 1e-8, 0.01, bc1, bc2)


def pow_f(b: Float64, n: Int) -> Float64:
    var r: Float64 = 1.0
    for _ in range(n):
        r *= b
    return r


def cos_f(x: Float64) -> Float64:
    return cos(x)


def median_of(v: List[Float64]) -> Float64:
    var n = len(v)
    if n == 0:
        return 0.0
    var w = List[Float64](length=n, fill=0.0)
    for i in range(n):
        w[i] = v[i]
    for i in range(1, n):
        var x = w[i]
        var j = i - 1
        while j >= 0 and w[j] > x:
            w[j + 1] = w[j]
            j -= 1
        w[j + 1] = x
    return w[n // 2]


# ---------------------------------------------------------------- data
struct Data(Movable):
    var n: Int
    var z: BufF
    var y: BufF
    var v: BufF
    var idx: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, name: String, n: Int, maxb: Int) raises:
        self.n = n
        self.z = load_f32(ctx, DATA + name + "_Z_" + String(CTX) + ".f32", n * CTX)
        self.y = load_f32(ctx, DATA + name + "_Y_" + String(CTX) + ".f32", n * 32)
        self.v = load_f32(ctx, DATA + name + "_V_" + String(CTX) + ".f32", n * 32)
        self.idx = ctx.enqueue_create_buffer[DType.int32](maxb)


def gather_batch(ctx: DeviceContext, d: Data, a: Acts, ids: List[Int], B: Int) raises:
    var hb = ctx.enqueue_create_host_buffer[DType.int32](B)
    for i in range(B):
        hb[i] = Int32(ids[i])
    ctx.enqueue_copy(d.idx, hb)
    var pi = d.idx.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var pz = pf(d.z)
    var py = pf(d.y)
    var pv = pf(d.v)
    var oz = pf(a.xin)
    var oy = pf(a.yb)
    var ov = pf(a.vb_)
    var B64 = Int64(B)

    def k() {var pi, var pz, var py, var pv, var oz, var oy, var ov, var B64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(B64) * CTX:
            var r = i // CTX
            var c = i % CTX
            oz[i] = pz[Int(pi[r]) * CTX + c]
        if i < Int(B64) * 32:
            var r = i // 32
            var c = i % 32
            oy[i] = py[Int(pi[r]) * 32 + c]
            ov[i] = pv[Int(pi[r]) * 32 + c]

    ctx.enqueue_function(k, grid_dim=ceildiv(B * CTX, 256), block_dim=256)


def eval_val(ctx: DeviceContext, net: Net, a: Acts, d: Data, B: Int) raises -> List[Float32]:
    """Returns [pinball loss, step-0 MAE of the median / MAE of zero, step-0 direction accuracy] over d."""
    var tot_loss: Float64 = 0
    var tot_w: Float64 = 0
    var mae: Float64 = 0
    var mae0: Float64 = 0
    var hit: Float64 = 0
    var cnt: Float64 = 0
    var nchunks = d.n // B
    for c in range(nchunks):
        var ids = List[Int](length=B, fill=0)
        for i in range(B):
            ids[i] = c * B + i
        gather_batch(ctx, d, a, ids, B)
        forward_train(ctx, net, a)
        var vh = host_f32(ctx, a.vb_, B * 32)
        var wsum: Float32 = 0
        for b in range(B):
            for k in range(32):
                var wk: Float32 = 0.5 / 31.0
                if k == 0:
                    wk = 1.0
                wsum += wk * vh[b * 32 + k]
        k_loss(ctx, a, net, Float32(1.0) / (wsum * Float32(NQ) * 0.02))
        ctx.synchronize()
        var lb = host_f32(ctx, a.lossb, B)
        var mb = host_f32(ctx, a.medb, B)
        var yh = host_f32(ctx, a.yb, B * 32)
        for b in range(B):
            tot_loss += Float64(lb[b])
            var y0 = Float64(yh[b * 32])
            var m0 = Float64(mb[b])
            mae += abs(y0 - m0)
            mae0 += abs(y0)
            if (y0 > 0) == (m0 > 0):
                hit += 1.0
            cnt += 1.0
        tot_w += Float64(wsum)
    var out = List[Float32](length=3, fill=0.0)
    out[0] = Float32(tot_loss / tot_w)
    out[1] = Float32(mae / mae0)
    out[2] = Float32(hit / cnt)
    return out^


def load_batch(ctx: DeviceContext, a: Acts, B: Int) raises:
    var zin = load_f32(ctx, DATA + "tb_Z_" + String(CTX) + ".f32", 64 * CTX)
    var yin = load_f32(ctx, DATA + "tb_Y_" + String(CTX) + ".f32", 64 * 32)
    var vin = load_f32(ctx, DATA + "tb_V_" + String(CTX) + ".f32", 64 * 32)
    ctx.enqueue_copy(a.xin, zin.create_sub_buffer[F32](0, B * CTX))
    ctx.enqueue_copy(a.yb, yin.create_sub_buffer[F32](0, B * 32))
    ctx.enqueue_copy(a.vb_, vin.create_sub_buffer[F32](0, B * 32))
    ctx.synchronize()


def batch_loss(ctx: DeviceContext, net: Net, a: Acts, B: Int, lscale: Float32) raises -> Float32:
    var vh = host_f32(ctx, a.vb_, B * 32)
    var wsum: Float32 = 0
    for b in range(B):
        for k in range(32):
            var wk: Float32 = 0.5 / 31.0
            if k == 0:
                wk = 1.0
            wsum += wk * vh[b * 32 + k]
    k_loss(ctx, a, net, lscale / (wsum * Float32(NQ) * 0.02))
    ctx.synchronize()
    var lb = host_f32(ctx, a.lossb, B)
    var tot: Float32 = 0
    for b in range(B):
        tot += lb[b]
    return tot / wsum


def main() raises:
    var args = argv()
    var mode = String(args[1])
    with DeviceContext() as ctx:
        if mode == "gradcheck":
            var B = Int(args[2])
            var net = Net(ctx)
            var a = Acts(ctx, B)
            load_batch(ctx, a, B)
            zero_grads(ctx, net)
            forward_train(ctx, net, a)
            var loss = batch_loss(ctx, net, a, B, 1.0)
            var tpg = TP(False)
            backward_train(ctx, net, a, tpg)
            ctx.synchronize()
            print("loss", loss)
            for i in range(len(net.P)):
                var g = host_f32(ctx, net.P[i].g, net.P[i].n)
                var f = open("mojo/data/train/grads_mojo_" + String(CTX) + "/" + net.P[i].name + ".f32", "w")
                var b = List[UInt8](length=net.P[i].n * 4, fill=0)
                var pb = b.unsafe_ptr().unsafe_bitcast[Float32]()
                for j in range(net.P[i].n):
                    pb[j] = g[j]
                f.write_bytes(Span(b))
                f.close()

        if mode == "train" or mode == "bench":
            var B = Int(args[2])
            var steps = Int(args[3])
            var lr_peak = Float32(Float64(String(args[4]))) if len(args) > 4 else Float32(2e-5)
            var eval_every = Int(args[5]) if len(args) > 5 else 100
            var ls = Float32(Float64(String(args[6]))) if len(args) > 6 else Float32(1.0)
            var net = Net(ctx)
            var a = Acts(ctx, B)
            var tr = Data(ctx, "train", 32768, B)
            var va = Data(ctx, "val", 4087, B)
            var idxf = open(DATA + "train_idx.i32", "r")
            var idxb = idxf.read_bytes()
            var idxp = idxb.unsafe_ptr().unsafe_bitcast[Int32]()
            var nidx = len(idxb) // 4
            var tp = TP(mode == "bench" and len(args) > 7)
            var warm = 100
            var t_train: Float64 = 0
            var t_fwd: Float64 = 0
            var t_min: Float64 = 1.0e30
            var tlist = List[Float64]()
            for step in range(1, steps + 1):
                var ids = List[Int](length=B, fill=0)
                for i in range(B):
                    ids[i] = Int(idxp[((step - 1) * B + i) % nidx])
                var lr = lr_peak
                if step < warm:
                    lr = lr_peak * Float32(step) / Float32(warm)
                else:
                    var prog = Float64(step - warm) / Float64(max(steps - warm, 1))
                    lr = lr_peak * Float32(0.5 * (1.0 + cos_f(3.141592653589793 * prog)))
                ctx.synchronize()
                var t0 = perf_counter_ns()
                gather_batch(ctx, tr, a, ids, B)
                var vh = host_f32(ctx, a.vb_, B * 32)
                var wsum: Float32 = 0
                for b in range(B):
                    for k in range(32):
                        var wk: Float32 = 0.5 / 31.0
                        if k == 0:
                            wk = 1.0
                        wsum += wk * vh[b * 32 + k]
                zero_grads(ctx, net)
                tp.lap(ctx, 0)
                forward_train(ctx, net, a)
                tp.lap(ctx, 1)
                k_loss(ctx, a, net, ls / (wsum * Float32(NQ) * 0.02))
                tp.lap(ctx, 2)
                backward_train(ctx, net, a, tp)
                optimizer_step(ctx, net, a, step, lr, ls, 1.0)
                tp.lap(ctx, 8)
                ctx.synchronize()
                var dts = Float64(perf_counter_ns() - t0)
                t_train += dts
                if step > 3:
                    tlist.append(dts)
                    t_min = min(t_min, dts)
                if mode == "train" and (step % eval_every == 0 or step == 1):
                    var lb = host_f32(ctx, a.lossb, B)
                    var tl: Float32 = 0
                    for b in range(B):
                        tl += lb[b]
                    var cl = host_f32(ctx, a.clip, 2)
                    var ev = eval_val(ctx, net, a, va, B)
                    print("step", step, "lr", lr, "train_loss", tl / wsum, "gnorm", cl[1], "| val loss", ev[0], "mae_vs_zero", ev[1], "dir", ev[2], "| ms/step", t_train / 1e6 / Float64(step))
            if tp.on:
                var tot: Float64 = 0
                for i in range(9):
                    tot += tp.acc[i]
                for i in range(9):
                    print("  ", tp_name(i), "ms/step", tp.acc[i] / 1e6 / Float64(steps), "pct", 100.0 * tp.acc[i] / tot)
            var med = median_of(tlist)
            print("RESULT mojo-train", String(DT), "B=" + String(B), "min", t_min / 1e6, "ms/step =", t_min / 1e3 / Float64(B), "us/sample; median", med / 1e6, "ms/step =", med / 1e3 / Float64(B), "us/sample")
