# Crypto LoRA Batch (smoke2)

- Total runs: 1
- Successful: 0
- Errors: 1

| Symbol | Preaug | Ctx | LR | Val MAE% | Test MAE% | Val Consistency | Output Dir |
|---|---|---:|---:|---:|---:|---:|---|

## Errors

- DOGEUSD percent_change ctx=128 lr=5e-05: ^^^^^^^^^
  File "/nvme0n1-disk/code/stock-prediction/.venv313/lib/python3.13/site-packages/transformers/modeling_utils.py", line 4725, in caching_allocator_warmup
    free_device_memory, total_device_memory = accelerator_module.mem_get_info(index)
                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/nvme0n1-disk/code/stock-prediction/.venv313/lib/python3.13/site-packages/torch/cuda/memory.py", line 838, in mem_get_info
    return torch.cuda.cudart().cudaMemGetInfo(device)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
torch.AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
