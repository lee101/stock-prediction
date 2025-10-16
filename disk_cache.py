import functools
import hashlib
import os
import pickle
import shutil
import time

import torch


def disk_cache(func):
    cache_dir = os.path.join(os.path.dirname(__file__), '.cache', func.__name__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we're in testing mode
        if os.environ.get('TESTING') == 'True':
            return func(*args, **kwargs)

        # Create a unique key based on the function arguments
        key_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg.detach().cpu().numpy() if hasattr(arg, "detach") else arg.cpu().numpy()
                key_parts.append(hashlib.md5(tensor.tobytes()).hexdigest())
            else:
                key_parts.append(str(arg))
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                tensor = v.detach().cpu().numpy() if hasattr(v, "detach") else v.cpu().numpy()
                key_parts.append(f"{k}:{hashlib.md5(tensor.tobytes()).hexdigest()}")
            else:
                key_parts.append(f"{k}:{v}")

        key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{key}.pkl')

        # Check if the result is already cached
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # If not cached, call the function and cache the result
        result = func(*args, **kwargs)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result

    def cache_clear():
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        time.sleep(0.1)  # Add a small delay to ensure the directory is removed
        os.makedirs(cache_dir, exist_ok=True)

    wrapper.cache_clear = cache_clear
    return wrapper
