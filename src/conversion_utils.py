import torch

def unwrap_tensor(data):
    if isinstance(data, torch.Tensor):
        if data.dim() == 0:
            return float(data)
        elif data.dim() >= 1:
            return data.tolist()
    else:
        return data