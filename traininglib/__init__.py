from .runtime_flags import enable_fast_kernels, bf16_supported
from .compile_wrap import maybe_compile
from .optim_factory import make_optimizer, MultiOptim
from .schedules import WarmupCosine
from .report import write_report_markdown
