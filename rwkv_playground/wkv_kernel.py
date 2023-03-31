from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn
from torch.utils.cpp_extension import load

"""This module is a wrapper for the C++/CUDA implementation of the WKV kernel.
"""


@dataclass
class WKVConfig:
    T_max: int = 1024  # max sequence length within cuda operations
    cpp_ext_name: str = "wkv"
    cpp_ext_sources: List[str] = field(
        default_factory=lambda: ["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"])
    extra_cuda_cflags: List[str] = field(default_factory=lambda: [
        '-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3',
        '-Xptxas -O3'
    ])
    device: str = "cuda"
    float_mode: str = "fp32"  # options: fp32, fp16, bfloat16

    def __post_init__(self):
        self.extra_cuda_cflags.append(f'-DTmax={self.T_max}')
        self.device = torch.device(self.device)

    def float_mode_to_dtype(self):
        if self.float_mode == "fp32":
            return torch.float32
        elif self.float_mode == "fp16":
            return torch.float16
        elif self.float_mode == "bfloat16":
            return torch.bfloat16
        else:
            raise ValueError(f"Unknown float_mode: {self.float_mode}")


class WKV(nn.Module):
    _instance = None # for singleton

    class _WKV(torch.autograd.Function):

        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v, wkv_cuda):
            # TODO add wkv_cuda to context
            pass

        @staticmethod
        def backward(ctx, gy):
            pass

    def __new__(cls, config: WKVConfig = WKVConfig()):
        if cls._instance is None:
            cls._instance = super(WKV, cls).__new__(cls)
            cls._instance._setup(config)
        return cls._instance
    
    def __init__(self, *args, **kwargs):
        # Dummy to avoid multiple calls to self._load_cuda()
        pass

    def _setup(self, config: WKVConfig = WKVConfig()):
        """Setup the WKV module. This is called by __new__ as constructor."""
        super().__init__()
        self.cfg = config
        self.wkv_cuda = self._load_cuda()
        self.device = self.cfg.device
    
    def _load_cuda(self):
        cfg = self.cfg
        cuda_module = load(name=cfg.cpp_ext_name,
                           sources=cfg.cpp_ext_sources,
                           verbose=True,
                           extra_cuda_cflags=cfg.extra_cuda_cflags)
        return cuda_module

    def to(self, **kwargs):
        device = kwargs.get('device', None)
        if device is not None:
            self.device = torch.device(device)
        return super().to(**kwargs)

    def forward(self, B, T, C, w, u, k, v):
        assert self.device != torch.device(
            "cpu"), "WKV is not implemented for CPU"
        return self._WKV.apply(B, T, C, w, u, k, v, self.wkv_cuda)
