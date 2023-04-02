import math
from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch import nn

from rwkv_cleanup.wkv_kernel import WKV, WKVConfig


@dataclass
class RWKVConfig:
    embedding_dim: int
    attention_dim: int
    ffn_dim: int
    num_blocks: int
    vocab_size: int
    context_len: int
    wkv_config: WKVConfig = field(default_factory=lambda: WKVConfig())


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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init embedding
        # default init is zero # TODO try this
        # we use a narrow uniform init, in the original code they use the initial learning rate
        # we just set it to a small value
        emb_init_range = 0.0008  #1e-3
        nn.init.uniform_(self.embedding.weight,
                         a=-emb_init_range,
                         b=emb_init_range)
        # init blocks
        for b in self.blocks:
            b.reset_parameters()
        # init head and layer norm
        self.head.reset_parameters()
        self.ln_out.reset_parameters()

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


def _calc_gain(weight: torch.Tensor) -> float:
    """Calculate the gain value of the given weight tensor."""
    gain = 1.0
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_out > fan_in:
        gain = math.sqrt(fan_out / fan_in)
    return gain


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

        embedding_dim = self.rwkv_cfg.embedding_dim
        attention_dim = self.rwkv_cfg.attention_dim
        # init time mix constants
        req_grad = True  # TODO make this configurable
        self.time_mix_k = nn.Parameter(torch.empty((1, 1, embedding_dim)),
                                       requires_grad=req_grad)
        self.time_mix_v = nn.Parameter(torch.empty((1, 1, embedding_dim)),
                                       requires_grad=req_grad)
        self.time_mix_r = nn.Parameter(torch.empty((1, 1, embedding_dim)),
                                       requires_grad=req_grad)

        # init time decay
        self.time_decay = nn.Parameter(torch.empty((attention_dim, )),
                                       requires_grad=req_grad)
        self.time_first = nn.Parameter(torch.empty((attention_dim, )),
                                       requires_grad=req_grad)

        # init layers / parameters
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.receptance = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, embedding_dim, bias=False)

        self.wkv = WKV(config=self.rwkv_cfg.wkv_config)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init time mix constants
        time_mix_k, time_mix_v, time_mix_r = self._init_time_mix_constants()
        req_grad = True
        self.time_mix_k = nn.Parameter(time_mix_k, requires_grad=req_grad)
        self.time_mix_v = nn.Parameter(time_mix_v, requires_grad=req_grad)
        self.time_mix_r = nn.Parameter(time_mix_r, requires_grad=req_grad)
        # init time decay
        time_decay, time_first = self._init_time_decay_constants()
        self.time_decay = nn.Parameter(time_decay, requires_grad=req_grad)
        self.time_first = nn.Parameter(time_first, requires_grad=req_grad)
        # init layers / parameters
        # ZERO INIT
        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.receptance.weight)
        nn.init.zeros_(self.output.weight)
        # ORTHOGONAL INIT
        nn.init.orthogonal_(self.value.weight,
                            gain=_calc_gain(self.value.weight))

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
        # wkv cuda kernel
        rwkv = sr * self.wkv(B, T, attention_dim, self.time_decay,
                             self.time_first, k, v)
        return self.output(rwkv)

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
        time_decay = decay_speed

        # time first # TODO does this make sense?
        zigzag = torch.tensor([(i + 1) % 3 - 1
                               for i in range(attention_dim)]) * 0.5
        time_first = torch.ones(attention_dim) * torch.log(
            torch.tensor(0.3)) + zigzag

        return time_decay, time_first


class RWKVChannelMix(nn.Module):

    def __init__(self, rwkv_config: RWKVConfig, block_id: int):
        super().__init__()
        self.rwkv_cfg = rwkv_config
        self.block_id = block_id

        embedding_dim = self.rwkv_cfg.embedding_dim
        ffn_dim = self.rwkv_cfg.ffn_dim
        # init time mix constants
        req_grad = True
        self.time_mix_k = nn.Parameter(torch.empty((1, 1, embedding_dim)),
                                       requires_grad=req_grad)
        self.time_mix_r = nn.Parameter(torch.empty((1, 1, embedding_dim)),
                                       requires_grad=req_grad)

        # init layers / parameters
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(embedding_dim, ffn_dim, bias=False)
        self.receptance = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(ffn_dim, embedding_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # init time mix constants
        time_mix_k, time_mix_r = self._init_time_mix_constants()
        req_grad = True
        self.time_mix_k = nn.Parameter(time_mix_k, requires_grad=req_grad)
        self.time_mix_r = nn.Parameter(time_mix_r, requires_grad=req_grad)
        # init layers / parameters
        # ZERO INIT
        nn.init.zeros_(self.receptance.weight)
        nn.init.zeros_(self.value.weight)
        # ORTHOGONAL INIT
        nn.init.orthogonal_(self.key.weight, gain=_calc_gain(self.key.weight))

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