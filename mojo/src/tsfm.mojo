"""Timestral-209M univariate one-step forward (context 512 log returns -> 21 quantiles) on the GPU.

Tensor-core GEMMs from the Modular kernel library; everything else is custom fused kernels. Residual stream is fp32,
GEMM operands/outputs are DT (bf16 or fp16). Math follows mojo/torch_fast.py, which matches the official model.
"""

from std.math import sqrt, exp, asinh, sinh, log, ceildiv
from std.memory import stack_allocation, unsafe_memcpy
from std.sys import argv, size_of
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from layout import TileTensor, row_major, Idx
from linalg.matmul.gpu import _matmul_gpu

comptime DT = DType.bfloat16
comptime EXT = "bf16" if DT == DType.bfloat16 else "f16"
comptime F32 = DType.float32
comptime D = 768
comptime NH = 12
comptime HD = 64
comptime FF = 2048
comptime S = 17
comptime L = 16
comptime NL = 24
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


def mm[N: Int, K: Int](ctx: DeviceContext, c: BufD, a: BufD, w: BufD, m: Int) raises:
    """c[m, N] = a[m, K] @ w[N, K]^T."""
    var tc = TileTensor(mpd(c), row_major(m, Idx[N]))
    var ta = TileTensor(mpd(a), row_major(m, Idx[K]))
    var tw = TileTensor(mpd(w), row_major(Idx[N], Idx[K]))
    _matmul_gpu[use_tensor_core=True, transpose_b=True](tc, ta, tw, ctx)


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


struct Model(Movable):
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
    # activations, sized for R rows
    var R: Int
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
    var d: BufD
    var hh: BufD
    var r2: BufD
    var rr: BufD
    var out: BufF

    def __init__(out self, ctx: DeviceContext, R: Int, nout: Int) raises:
        self.layers = List[Layer]()
        for i in range(NL):
            self.layers.append(Layer(ctx, i, i % 4 == 3))
        self.fq = load[F32](ctx, DATA + "fq.f32", D)
        self.e1b = load[F32](ctx, DATA + "e1b.f32", 4 * D)
        self.e2b = load[F32](ctx, DATA + "e2b.f32", D)
        self.erb = load[F32](ctx, DATA + "erb.f32", D)
        self.on = load[F32](ctx, DATA + "on.f32", D)
        self.cosb = load[F32](ctx, DATA + "cos.f32", S * HD)
        self.sinb = load[F32](ctx, DATA + "sin.f32", S * HD)
        self.qw = load[F32](ctx, DATA + "qw.f32", 21)
        self.e1 = load[DT](ctx, DATA + "e1." + EXT, 4 * D * 32)
        self.e2 = load[DT](ctx, DATA + "e2." + EXT, D * 4 * D)
        self.er = load[DT](ctx, DATA + "er." + EXT, D * 32)
        self.h1 = load[DT](ctx, DATA + "h1." + EXT, 4 * D * D)
        self.h2 = load[DT](ctx, DATA + "h2." + EXT, 32 * 4 * D)
        self.hr = load[DT](ctx, DATA + "hr." + EXT, 32 * D)
        self.R = R
        var M = R * S
        self.xin = ctx.enqueue_create_buffer[F32](R * 512)
        self.z = ctx.enqueue_create_buffer[DT](R * L * 32)
        self.loc = ctx.enqueue_create_buffer[F32](R * L)
        self.sc = ctx.enqueue_create_buffer[F32](R * L)
        self.e1h = ctx.enqueue_create_buffer[DT](R * L * 4 * D)
        self.e2o = ctx.enqueue_create_buffer[DT](R * L * D)
        self.ero = ctx.enqueue_create_buffer[DT](R * L * D)
        self.x = ctx.enqueue_create_buffer[F32](M * D)
        self.h = ctx.enqueue_create_buffer[DT](M * D)
        self.big = ctx.enqueue_create_buffer[DT](M * 2 * FF)
        self.o = ctx.enqueue_create_buffer[DT](M * D)
        self.t = ctx.enqueue_create_buffer[DT](M * D)
        self.u = ctx.enqueue_create_buffer[DT](M * FF)
        self.d = ctx.enqueue_create_buffer[DT](R * D)
        self.hh = ctx.enqueue_create_buffer[DT](R * 4 * D)
        self.r2 = ctx.enqueue_create_buffer[DT](R * 32)
        self.rr = ctx.enqueue_create_buffer[DT](R * 32)
        self.out = ctx.enqueue_create_buffer[F32](nout * 21)


def k_prep(ctx: DeviceContext, m: Model, nb: Int) raises:
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
        var row = px + src * 512
        var cumx: Float64 = 0
        var mean: Float64 = 0
        var m2: Float64 = 0
        for t in range(512):
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


def k_silu_bias(ctx: DeviceContext, m: Model, rows: Int) raises:
    var p = mpd(m.e1h)
    var pb = mpf(m.e1b)
    var tot64 = Int64(rows * 4 * D)

    def k() {var p, var pb, var tot64}:
        var tot = Int(tot64)
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < tot:
            var v = Float32(p[i]) + pb[i % (4 * D)]
            p[i] = Scalar[DT](v / (1.0 + exp(-v)))

    ctx.enqueue_function(k, grid_dim=ceildiv(Int(tot64), 256), block_dim=256)


def k_embed(ctx: DeviceContext, m: Model, R: Int, n0: BufF) raises:
    """x[b, p] = e2 + b2 + er + erb (p < 16), x[b, 16] = forecast query; h = rmsnorm(x) * n0."""
    var px = mpf(m.x)
    var ph = mpd(m.h)
    var pe = mpd(m.e2o)
    var pr = mpd(m.ero)
    var pb2 = mpf(m.e2b)
    var pbr = mpf(m.erb)
    var pfq = mpf(m.fq)
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


def k_add_rms(ctx: DeviceContext, m: Model, M: Int, w: BufF) raises:
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


def k_attn(ctx: DeviceContext, m: Model, R: Int) raises:
    """Causal attention over the 17 tokens of each (row, head) with RoPE and the sigmoid gate; reads big = [q|k|v|gate].
    One block per (row, head): 288 threads; scores one thread per (query, key) pair, softmax per query row, PV per output."""
    var pq = mpd(m.big)
    var po = mpd(m.o)
    var pc = mpf(m.cosb)
    var ps = mpf(m.sinb)
    comptime NT = 288
    comptime KS = 65

    def k() {var pq, var po, var pc, var ps}:
        var h = Int(block_idx.x) % NH
        var b = Int(block_idx.x) // NH
        var t = Int(thread_idx.x)
        var qs = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var ks = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var vs = stack_allocation[S * KS, Float32, address_space=AddressSpace.SHARED]()
        var sm = stack_allocation[S * S, Float32, address_space=AddressSpace.SHARED]()
        for idx in range(t, S * HD, NT):
            var j = idx // HD
            var d = idx % HD
            var row = pq + (b * S + j) * (4 * D) + h * HD
            var pd = d + 32
            if d >= 32:
                pd = d - 32
            var c = pc[j * HD + d]
            var s = ps[j * HD + d]
            var qd = Float32(row[d])
            var qp = Float32(row[pd])
            var kd = Float32(row[D + d])
            var kp = Float32(row[D + pd])
            if d < 32:
                qs[j * KS + d] = qd * c - qp * s
                ks[j * KS + d] = kd * c - kp * s
            else:
                qs[j * KS + d] = qd * c + qp * s
                ks[j * KS + d] = kd * c + kp * s
            vs[j * KS + d] = Float32(row[2 * D + d])
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
        for idx in range(t, S * HD, NT):
            var qi = idx // HD
            var d = idx % HD
            var acc: Float32 = 0
            for j in range(qi + 1):
                acc += sm[qi * S + j] * vs[j * KS + d]
            var g = Float32(pq[(b * S + qi) * (4 * D) + 3 * D + h * HD + d])
            var gs = Float32(1.0) / (Float32(1.0) + exp(-g))
            po[(b * S + qi) * D + h * HD + d] = Scalar[DT](acc * gs)

    ctx.enqueue_function(k, grid_dim=R * NH, block_dim=NT)


def k_vgate(ctx: DeviceContext, m: Model, M: Int) raises:
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


def k_swiglu(ctx: DeviceContext, m: Model, M: Int) raises:
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


def k_final_norm(ctx: DeviceContext, m: Model, R: Int) raises:
    """d[b] = rmsnorm(x[b, 16]) * out_norm."""
    var px = mpf(m.x)
    var pd = mpd(m.d)
    var pw = mpf(m.on)

    def k() {var px, var pd, var pw}:
        var b = Int(block_idx.x)
        var t = Int(thread_idx.x)
        var row = px + (b * S + (S - 1)) * D
        var v0 = row[t]
        var v1 = row[t + 256]
        var v2 = row[t + 512]
        var s = v0 * v0 + v1 * v1 + v2 * v2
        var tot = block_sum[block_size=256](s)
        var inv = Float32(1.0) / sqrt(tot / Float32(D) + 1e-6)
        pd[b * D + t] = Scalar[DT](v0 * inv * pw[t])
        pd[b * D + t + 256] = Scalar[DT](v1 * inv * pw[t + 256])
        pd[b * D + t + 512] = Scalar[DT](v2 * inv * pw[t + 512])

    ctx.enqueue_function(k, grid_dim=R, block_dim=256)


def k_silu(ctx: DeviceContext, m: Model, n_: Int) raises:
    var p = mpd(m.hh)
    var n64 = Int64(n_)

    def k() {var p, var n64}:
        var n = Int(n64)
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < n:
            var v = Float32(p[i])
            p[i] = Scalar[DT](v / (Float32(1.0) + exp(-v)))

    ctx.enqueue_function(k, grid_dim=ceildiv(n_, 256), block_dim=256)


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


def k_head_post(ctx: DeviceContext, m: Model, nout: Int, sa: Bool) raises:
    var pr2 = mpd(m.r2)
    var prr = mpd(m.rr)
    var pqw = mpf(m.qw)
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


def forward(ctx: DeviceContext, m: Model, nb: Int, sa: Bool, mut pf: Prof) raises:
    var R = m.R
    var M = R * S
    pf.last = Int(perf_counter_ns())
    k_prep(ctx, m, nb)
    mm[4 * D, 32](ctx, m.e1h, m.z, m.e1, R * L)
    k_silu_bias(ctx, m, R * L)
    mm[D, 4 * D](ctx, m.e2o, m.e1h, m.e2, R * L)
    mm[D, 32](ctx, m.ero, m.z, m.er, R * L)
    k_embed(ctx, m, R, m.layers[0].n0)
    pf.lap(ctx, 0)
    for i in range(NL):
        ref ly = m.layers[i]
        mm[4 * D, D](ctx, m.big, m.h, ly.qkvg, M)
        pf.lap(ctx, 1)
        k_attn(ctx, m, R)
        pf.lap(ctx, 2)
        mm[D, D](ctx, m.t, m.o, ly.o0, M)
        pf.lap(ctx, 3)
        if ly.variate:
            k_add_rms(ctx, m, M, ly.n1)
            pf.lap(ctx, 4)
            mm[2 * D, D](ctx, m.big, m.h, ly.vg, M)
            pf.lap(ctx, 5)
            k_vgate(ctx, m, M)
            pf.lap(ctx, 6)
            mm[D, D](ctx, m.t, m.o, ly.o1, M)
            pf.lap(ctx, 3)
        k_add_rms(ctx, m, M, ly.nm)
        pf.lap(ctx, 4)
        mm[2 * FF, D](ctx, m.big, m.h, ly.w12, M)
        pf.lap(ctx, 7)
        k_swiglu(ctx, m, M)
        pf.lap(ctx, 8)
        mm[D, FF](ctx, m.t, m.u, ly.w3, M)
        pf.lap(ctx, 9)
        if i + 1 < NL:
            k_add_rms(ctx, m, M, m.layers[i + 1].n0)
        else:
            k_add_rms(ctx, m, M, m.on)
        pf.lap(ctx, 4)
    k_final_norm(ctx, m, R)
    mm[4 * D, D](ctx, m.hh, m.d, m.h1, R)
    k_silu(ctx, m, R * 4 * D)
    mm[32, 4 * D](ctx, m.r2, m.hh, m.h2, R)
    mm[32, D](ctx, m.rr, m.d, m.hr, R)
    k_head_post(ctx, m, nb, sa)
    pf.lap(ctx, 10)


def main() raises:
    var args = argv()
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
        var m = Model(ctx, R, B)
        var xin = load[F32](ctx, DATA + "test_in.f32", 1024 * 512)
        ctx.enqueue_copy(m.xin, xin)   # first R*512 floats; mirrored rows are derived in k_prep
        ctx.synchronize()
        var pf0 = Prof(False)
        forward(ctx, m, B, sa, pf0)
        ctx.synchronize()
        var host = ctx.enqueue_create_host_buffer[F32](B * 21)
        ctx.enqueue_copy(host, m.out)
        ctx.synchronize()
        var ref_all = load[F32](ctx, DATA + "test_ref_sa" + String(1 if sa else 0) + ".f32", 1024 * 21)
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
            forward(ctx, m, B, sa, pf0)
        ctx.synchronize()
        var dt = Float64(perf_counter_ns() - t0) / Float64(iters)
        print("us/window", dt / 1e3 / Float64(B), " ms/batch", dt / 1e6)
        var pf = Prof(True)
        for _ in range(iters):
            forward(ctx, m, B, sa, pf)
        var tot: Float64 = 0
        for i in range(11):
            tot += pf.acc[i]
        for i in range(11):
            print("  ", pname(i), "us/win", pf.acc[i] / Float64(iters) / 1e3 / Float64(B), "pct", 100.0 * pf.acc[i] / tot)
