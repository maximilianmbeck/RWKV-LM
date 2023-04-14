from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import torch
from einops import rearrange, repeat
from torch import nn
from torch.utils.cpp_extension import load

"""This module is a wrapper for the C++/CUDA implementation of the WKV kernel.
"""


@dataclass
class WKVConfig:
    T_max: int = 1024  # max sequence length within cuda operations
    cpp_ext_name: str = 'wkv'
    cpp_ext_sources: List[str] = field(default_factory=lambda: [
        str(Path(__file__).parent / 'cuda/wkv_op.cpp'),
        str(Path(__file__).parent / 'cuda/wkv_cuda.cu')
    ])
    extra_cuda_cflags: List[str] = field(default_factory=lambda: [
        '-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3',
        '-Xptxas -O3'
    ])
    device: Union[str, torch.device] = 'cuda'
    float_mode: str = 'fp32'  # options: fp32, fp16, bfloat16

    def __post_init__(self):
        self.extra_cuda_cflags.append(f'-DTmax={self.T_max}')
        self.device = torch.device(self.device)

    def float_mode_to_dtype(self):
        if self.float_mode == 'fp32' or '32' in str(self.float_mode):
            return torch.float32
        elif self.float_mode == 'fp16' or '16' in str(self.float_mode):
            return torch.float16
        elif self.float_mode == 'bfloat16':
            return torch.bfloat16
        else:
            raise ValueError(f'Unknown float_mode: {self.float_mode}')


class WKV(nn.Module):
    _instance = None  # for singleton

    class _WKV(torch.autograd.Function):

        wkv_cuda = None
        wkv_config: WKVConfig = None

        @classmethod
        def forward(cls, ctx, batch_size, seq_len, embedding_dim, time_decay,
                    time_first, k, v):
            wkv_cuda = cls.wkv_cuda
            wkv_config = cls.wkv_config
            # setup context # TODO for PyTorch 2.0 use extra setup_context() function
            ctx.batch_size = batch_size
            ctx.seq_len = seq_len
            ctx.embedding_dim = embedding_dim
            ctx.wkv_cuda = wkv_cuda
            ctx.wkv_config = wkv_config
            assert seq_len <= wkv_config.T_max, f'Sequence length {seq_len} exceeds the maximum allowed T_max={wkv_config.T_max}'
            # TODO what does this assert do? Why necessary?
            assert batch_size * embedding_dim % min(
                embedding_dim, wkv_config.T_max
            ) == 0, 'batch_size * embedding_dim must be divisible by min(embedding_dim, T_max)'
            #
            dtype = torch.float32  # convert all tensors to float32 (for cuda kernel)
            device = wkv_config.device
            # convert input tensors
            time_decay = time_decay.to(dtype=dtype,
                                       device=device,
                                       memory_format=torch.contiguous_format)
            time_first = time_first.to(dtype=dtype,
                                       device=device,
                                       memory_format=torch.contiguous_format)
            k = k.to(dtype=dtype,
                     device=device,
                     memory_format=torch.contiguous_format)
            v = v.to(dtype=dtype,
                     device=device,
                     memory_format=torch.contiguous_format)

            # allocate output tensor
            y = torch.empty(batch_size,
                            seq_len,
                            embedding_dim,
                            dtype=dtype,
                            device=device,
                            memory_format=torch.contiguous_format)

            # call cuda kernel
            time_decay = -torch.exp(time_decay)  # TODO why is this necessary?
            wkv_cuda.forward(batch_size, seq_len, embedding_dim, time_decay,
                             time_first, k, v, y)
            ctx.save_for_backward(time_decay, time_first, k, v, y)

            # convert output tensor to correct dtype
            y = y.to(dtype=wkv_config.float_mode_to_dtype())
            return y

        @staticmethod
        def backward(ctx, gy):
            batch_size = ctx.batch_size
            seq_len = ctx.seq_len
            embedding_dim = ctx.embedding_dim
            assert seq_len <= ctx.wkv_config.T_max, f'Sequence length {seq_len} exceeds the maximum allowed T_max={ctx.wkv_config.T_max}'
            assert batch_size * embedding_dim % min(
                embedding_dim, ctx.wkv_config.T_max
            ) == 0, 'batch_size * embedding_dim must be divisible by min(embedding_dim, T_max)'

            time_decay, time_first, k, v, y = ctx.saved_tensors

            device = ctx.wkv_config.device
            # allocate gradient tensors
            gtime_decay = torch.zeros((batch_size, embedding_dim),
                                      device=device,
                                      dtype=torch.float32).contiguous()
            gtime_first = torch.zeros((batch_size, embedding_dim),
                                      device=device,
                                      dtype=torch.float32).contiguous()
            gk = torch.zeros((batch_size, seq_len, embedding_dim),
                             device=device,
                             dtype=torch.float32).contiguous()
            gv = torch.zeros((batch_size, seq_len, embedding_dim),
                             device=device,
                             dtype=torch.float32).contiguous()

            # call cuda kernel
            gy = gy.to(dtype=torch.float32,
                       memory_format=torch.contiguous_format)
            # arg0: int, arg1: int, arg2: int, arg3: at::Tensor, arg4: at::Tensor, arg5: at::Tensor, arg6: at::Tensor,
            # arg7: at::Tensor, arg8: at::Tensor, arg9: at::Tensor, arg10: at::Tensor, arg11: at::Tensor, arg12: at::Tensor
            ctx.wkv_cuda.backward(batch_size, seq_len, embedding_dim,
                                  time_decay, time_first, k, v, y, gy,
                                  gtime_decay, gtime_first, gk, gv)

            gtime_decay = gtime_decay.sum(dim=0)
            gtime_first = gtime_first.sum(dim=0)

            # convert gradient tensors to correct dtype
            out_dtype = ctx.wkv_config.float_mode_to_dtype()

            return (None, None, None, gtime_decay.to(dtype=out_dtype),
                    gtime_first.to(dtype=out_dtype), gk.to(dtype=out_dtype),
                    gv.to(dtype=out_dtype))

    def __new__(cls, config: WKVConfig = WKVConfig()):
        if cls._instance is None:
            cls._instance = super(WKV, cls).__new__(cls)
            cls._instance._setup(config)
        return cls._instance

    def __init__(self, *args, **kwargs):
        # Dummy to avoid multiple calls to self._load_cuda()
        pass

    def _setup(self, config: WKVConfig = WKVConfig()):
        '''Setup the WKV module. This is called by __new__ as constructor.'''
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
            self.device = self.cfg.device = torch.device(device)
        return super().to(**kwargs)

    def forward(self, batch_size, seq_len, embeding_dim, time_decay,
                time_first, k, v):
        self.device = self.cfg.device = v.device
        assert self.device != torch.device(
            'cpu'), 'WKV is not implemented for CPU'
        self._WKV.wkv_cuda = self.wkv_cuda
        self._WKV.wkv_config = self.cfg
        return self._WKV.apply(batch_size, seq_len, embeding_dim, time_decay,
                               time_first, k, v)


class WKVTorch(nn.Module):

    def __init__(self):
        super().__init__()
        # did not add time_decay = -torch.exp(time_decay) here, it is not necessary
        # this is done in the forward function of WKV cuda kernel
        # if it is added here, the outputs will be different from the cuda kernel

    def forward(self, batch_size, seq_len, embedding_dim, time_decay,
                time_first, k, v):
        dtype = k.dtype
        device = k.device
        y = torch.zeros(batch_size,
                        seq_len,
                        embedding_dim,
                        dtype=dtype,
                        device=device)
        MIN_VAL = 0.0  #-1e38
        # reshape inputs
        k_ = rearrange(k, 'b s e -> s b e')
        v_ = rearrange(v, 'b s e -> s b e')
        y_ = rearrange(y, 'b s e -> s b e')
        tf = repeat(time_first, 'e -> b e', b=batch_size)
        td = repeat(time_decay, 'e -> b e', b=batch_size)
        # running sums
        aa = torch.zeros(batch_size, embedding_dim, dtype=dtype, device=device)
        bb = torch.zeros(batch_size, embedding_dim, dtype=dtype, device=device)
        eps = torch.full((batch_size, embedding_dim),
                         MIN_VAL,
                         dtype=dtype,
                         device=device)
        for t in range(seq_len):
            e_tf_k = torch.exp(tf + k_[t] - eps)
            y_[t] = (aa + v_[t] * e_tf_k) / (bb + e_tf_k)
            eps_next = torch.max(td + eps, k_[t])

            e_td_k = torch.exp(td + eps - eps_next)
            e_k = torch.exp(k_[t] - eps_next)
            aa = (aa * e_td_k + v_[t] * e_k)
            bb = (bb * e_td_k + e_k)
            eps = eps_next
        y = rearrange(y_, 's b e -> b s e')
        return y