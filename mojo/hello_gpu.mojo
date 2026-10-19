from max.gpu.host import DeviceContext
from max.gpu import thread_idx, block_idx, block_dim

comptime FP = Pointer[Float32, MutAnyOrigin]

@always_inline
def k_add(x: FP, n: Int32):
    var i = Int32(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        x[Int(i)] += 1.0

def main() raises:
    var ctx = DeviceContext()
    print("gpu:", ctx.name())
    var n = 1024
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        h[i] = Float32(i)
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(d, h)
    var p = d.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[k_add](p, Int32(n), grid_dim=(n + 255) // 256, block_dim=256)
    ctx.enqueue_copy(h, d)
    ctx.synchronize()
    print(h[0], h[1023])
