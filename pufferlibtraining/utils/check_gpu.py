#!/usr/bin/env python3
import torch

def main():
    print('CUDA available:', torch.cuda.is_available())
    print('torch:', torch.__version__, ' cuda:', torch.version.cuda)
    print('device count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'[{i}]', torch.cuda.get_device_name(i))

if __name__ == '__main__':
    main()

