import os
from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch import nn
from wkv_kernel import WKV, WKVConfig


@dataclass
class RWKVConfig:
    embedding_dim: int
    attention_dim: int
    ffn_dim: int
    num_blocks: int
    vocab_size: int
    context_len: int
    wkv_config: WKVConfig = field(default_factory=lambda: WKVConfig())
    # TODO make bias configurable

# TODO from here, create pytorch lightning module wrapper

class RWKV(nn.Module):

    def __init__(self, config: RWKVConfig, **kwargs):
        super().__init__()
        self.cfg = config

        self.embedding = nn.Embedding(num_embeddings=self.cfg.vocab_size,
                                      embedding_dim=self.cfg.embedding_dim)

        self.blocks = nn.ModuleList([
            RWKVBlock(rwkv_config=self.cfg, block_id=i)
            for i in range(self.cfg.num_blocks)
        ])

        self.ln_out = nn.LayerNorm(self.cfg.embedding_dim)
        self.head = nn.Linear(self.cfg.embedding_dim,
                              self.cfg.vocab_size,
                              bias=False)

    def forward(self, x):
        # input shape: (B, T), T <= context_len, T are token ids
        B, T = x.size()
        assert T <= self.cfg.context_len, f"input sequence length {T} exceeds context length {self.cfg.context_len}"

        x = self.embedding(x)  # (B, T, C), C = embedding_dim

        for i, block in enumerate(self.blocks):
            x = block(x)

        x = self.ln_out(x)
        x = self.head(x)
        return x


#------------------------------------------------
#                CUDA Kernel
#------------------------------------------------

# # copy from RWKV-v4neo

# T_MAX = 1024  # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# # it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

# from torch.utils.cpp_extension import load

# wkv_cuda = load(name="wkv",
#                 sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
#                 verbose=True,
#                 extra_cuda_cflags=[
#                     '-res-usage', '--maxrregcount 60', '--use_fast_math',
#                     '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'
#                 ])

# # TODO: make a class and pass arguments to the constructor
# class WKV(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, B, T, C, w, u, k, v):
#         ctx.B = B
#         ctx.T = T
#         ctx.C = C
#         assert T <= T_MAX
#         assert B * C % min(C, 1024) == 0
#         if '32' in os.environ['RWKV_FLOAT_MODE']:
#             w = -torch.exp(w.contiguous())
#             u = u.contiguous()
#             k = k.contiguous()
#             v = v.contiguous()
#         else:
#             w = -torch.exp(w.float().contiguous())
#             u = u.float().contiguous()
#             k = k.float().contiguous()
#             v = v.float().contiguous()
#         ctx.save_for_backward(w, u, k, v)
#         y = torch.empty((B, T, C),
#                         device='cuda',
#                         memory_format=torch.contiguous_format)
#         wkv_cuda.forward(B, T, C, w, u, k, v, y)
#         if '32' in os.environ['RWKV_FLOAT_MODE']:
#             return y
#         elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
#             return y.half()
#         elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
#             return y.bfloat16()

#     @staticmethod
#     def backward(ctx, gy):
#         B = ctx.B
#         T = ctx.T
#         C = ctx.C
#         assert T <= T_MAX
#         assert B * C % min(C, 1024) == 0
#         w, u, k, v = ctx.saved_tensors
#         gw = torch.zeros((B, C), device='cuda').contiguous()
#         gu = torch.zeros((B, C), device='cuda').contiguous()
#         gk = torch.zeros((B, T, C), device='cuda').contiguous()
#         gv = torch.zeros((B, T, C), device='cuda').contiguous()
#         if '32' in os.environ['RWKV_FLOAT_MODE']:
#             wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk,
#                               gv)
#         else:
#             wkv_cuda.backward(B, T, C, w, u, k, v,
#                               gy.float().contiguous(), gw, gu, gk, gv)
#         gw = torch.sum(gw, dim=0)
#         gu = torch.sum(gu, dim=0)
#         if '32' in os.environ['RWKV_FLOAT_MODE']:
#             return (None, None, None, gw, gu, gk, gv)
#         elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
#             return (None, None, None, gw.half(), gu.half(), gk.half(),
#                     gv.half())
#         elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
#             return (None, None, None, gw.bfloat16(), gu.bfloat16(),
#                     gk.bfloat16(), gv.bfloat16())

# def RUN_CUDA(B, T, C, w, u, k, v):
#     return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

#------------------------------------------------


class RWKVBlock(nn.Module):

    def __init__(self, rwkv_config: RWKVConfig, block_id: int):
        super().__init__()
        self.rwkv_cfg = rwkv_config
        self.block_id = block_id

        self.ln0 = None
        if self.block_id == 0:
            self.ln0 = nn.LayerNorm(self.rwkv_cfg.embedding_dim)
            # TODO 1) maybe additional positional embedding here (only in block 0)

        self.ln1 = nn.LayerNorm(self.rwkv_cfg.embedding_dim)
        self.ln2 = nn.LayerNorm(self.rwkv_cfg.embedding_dim)

        # TODO 2) maybe pre feedforward here (channel mix) see line 325f in RWKV-v4neo/model.py
        self.attention_timemix = RWKVTimeMix(rwkv_config=self.rwkv_cfg,
                                             block_id=self.block_id)
        self.ffn_channelmix = RWKVChannelMix(rwkv_config=self.rwkv_cfg,
                                             block_id=self.block_id)

    def reset_parameters(self) -> None:
        if self.ln0 is not None:
            self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.attention_timemix.reset_parameters()
        self.ffn_channelmix.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_id == 0 and self.ln0 is not None:
            x = self.ln0(x)
            # TODO 1) maybe positional embedding here (only in block 0)
            # x = x+pos_emb

        # TODO 2) maybe pre feedforward here (channel mix) see line 325f in RWKV-v4neo/model.py
        # residual connection 1
        x_ = self.ln1(x)
        x_ = self.attention_timemix(x_)
        x = x + x_
        # residual connection 2
        x_ = self.ln2(x)
        x_ = self.ffn_channelmix(x_)
        x = x + x_
        return x


class RWKVTimeMix(nn.Module):

    def __init__(self, rwkv_config: RWKVConfig, block_id: int):
        super().__init__()
        self.rwkv_cfg = rwkv_config
        self.block_id = block_id

        # init time mix constants
        time_mix_k, time_mix_v, time_mix_r = self._init_time_mix_constants()
        self.register_buffer('time_mix_k', time_mix_k, persistent=True)
        self.register_buffer('time_mix_v', time_mix_v, persistent=True)
        self.register_buffer('time_mix_r', time_mix_r, persistent=True)

        # init time decay
        time_decay, time_first = self._init_time_decay_constants()
        self.register_buffer('time_decay', time_decay, persistent=True)
        self.register_buffer('time_first', time_first, persistent=True)

        # init layers / parameters
        embedding_dim = self.rwkv_cfg.embedding_dim
        attention_dim = self.rwkv_cfg.attention_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.receptance = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, embedding_dim, bias=False)

        self.wkv = WKV(config=self.rwkv_cfg.wkv_config)

    def _compute_rkv(self, x):
        xx = self.time_shift(
            x)  # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)
        attention_dim = self.rwkv_cfg.attention_dim
        sr, k, v = self._compute_rkv(x)
        # rwkv = sr * RUN_CUDA(B, T, attention_dim, self.time_decay,
        #                      self.time_first, k, v)
        # use own implementation of WKV
        rwkv = sr * self.wkv(B, T, attention_dim, self.time_decay,
                             self.time_first, k, v)
        return self.output(rwkv)

    def reset_parameters(self):
        raise NotImplementedError

    def _init_time_mix_constants(
            self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_blocks = self.rwkv_cfg.num_blocks
        embedding_dim = self.rwkv_cfg.embedding_dim

        ratio_0_to_1 = self.block_id / (num_blocks - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (self.block_id / num_blocks)  # 1 to ~0

        # TODO does this make sense?
        # different time mix constants for each block and each embedding dim
        embed_dim_val = torch.ones(1, 1, embedding_dim)
        for i in range(embedding_dim):
            embed_dim_val[0, 0, i] = i / embedding_dim

        # TODO check constants 0.3 and 0.5
        time_mix_k = torch.pow(embed_dim_val, ratio_1_to_almost0)
        time_mix_v = torch.pow(embed_dim_val,
                               ratio_1_to_almost0) + 0.3 * ratio_0_to_1
        time_mix_r = torch.pow(embed_dim_val, 0.5 * ratio_1_to_almost0)

        return time_mix_k, time_mix_v, time_mix_r

    def _init_time_decay_constants(self) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = self.rwkv_cfg.num_blocks
        attention_dim = self.rwkv_cfg.attention_dim
        ratio_0_to_1 = self.block_id / (num_blocks - 1)  # 0 to 1

        # time decay
        decay_speed = torch.ones(attention_dim)
        for h in range(attention_dim):
            decay_speed[h] = -5 + 8 * (h / (attention_dim - 1))**(
                0.7 + 1.3 * ratio_0_to_1)
        time_decay = nn.Parameter(decay_speed)

        # time first # TODO does this make sense?
        zigzag = torch.tensor([(i + 1) % 3 - 1
                               for i in range(attention_dim)]) * 0.5
        time_first = nn.Parameter(
            torch.ones(attention_dim) * torch.log(torch.tensor(0.3)) + zigzag)

        return time_decay, time_first


class RWKVChannelMix(nn.Module):

    def __init__(self, rwkv_config: RWKVConfig, block_id: int):
        super().__init__()
        self.rwkv_cfg = rwkv_config
        self.block_id = block_id

        # init time mix constants
        time_mix_k, time_mix_r = self._init_time_mix_constants()
        self.register_buffer('time_mix_k', time_mix_k, persistent=True)
        self.register_buffer('time_mix_r', time_mix_r, persistent=True)

        # init layers / parameters
        embedding_dim = self.rwkv_cfg.embedding_dim
        ffn_dim = self.rwkv_cfg.ffn_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(embedding_dim, ffn_dim, bias=False)
        self.receptance = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(ffn_dim, embedding_dim, bias=False)

    def reset_parameters(self):
        raise NotImplementedError

    def _init_time_mix_constants(self) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = self.rwkv_cfg.num_blocks
        embedding_dim = self.rwkv_cfg.embedding_dim

        ratio_1_to_almost0 = 1.0 - (self.block_id / num_blocks)  # 1 to ~0
        embed_dim_val = torch.ones(1, 1, embedding_dim)
        for i in range(embedding_dim):
            embed_dim_val[0, 0, i] = i / embedding_dim

        time_mix_k = torch.pow(embed_dim_val, ratio_1_to_almost0)
        time_mix_r = torch.pow(embed_dim_val, ratio_1_to_almost0)

        return time_mix_k, time_mix_r

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        y = torch.sigmoid(self.receptance(xr)) * kv
        return y