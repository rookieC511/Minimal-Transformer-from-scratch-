# multi_head_attention.py
"""
一个最小可运行的 Transformer 实现（带改进）：
- 更鲁棒的 mask 构造
- Attention Dropout
- 保证 mask / 张量在正确的 device 上
- 可以返回注意力权重，便于可视化/调试
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    位置编码（Positional Encoding）
    使用 sin/cos 周期函数为序列位置编码，解决 Transformer 无法感知顺序的问题
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 构造 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 序号 (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，batch 维度
        self.register_buffer("pe", pe)  # register_buffer 不作为可训练参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # 将位置编码加到 embedding 上
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力（Multi-Head Attention）
    核心思想：并行多个注意力头，学习不同的子空间表示
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q、K、V 的线性变换
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(p=dropout)   # 对注意力分布做 dropout
        self.proj_dropout = nn.Dropout(p=dropout)   # 对最终输出做 dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (batch, seq_len, d_model)
        # 输出: (batch, n_heads, seq_len, d_k)
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (batch, n_heads, seq_len, d_k)
        # 输出: (batch, seq_len, d_model)
        batch, n_heads, seq_len, d_k = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, n_heads * d_k)
        return x

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled Dot-Product Attention
        q,k,v: (batch, n_heads, seq_len, d_k)
        mask: 用于屏蔽 padding 或未来信息，True 表示允许，False 表示屏蔽
        返回: (加权结果, 注意力权重)
        """
        # 打分矩阵
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, n_heads, q_len, k_len)

        if mask is not None:
            mask = mask.bool()
            # mask == False 的位置用 -inf 屏蔽，使 softmax≈0
            # 注意 mask 需要 broadcast 到 scores 的形状 (B, n_heads, q_len, k_len)
            scores = scores.masked_fill(~mask, float("-1e9"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

    def forward(
        self,
        q_input: torch.Tensor,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # QKV 线性变换
        q = self.q_linear(q_input)
        k = self.k_linear(k_input)
        v = self.v_linear(v_input)

        # 拆分成多头
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 注意力计算
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask=mask)

        # 合并多头
        concatenated = self._combine_heads(attn_output)
        output = self.out_linear(concatenated)
        output = self.proj_dropout(output)

        if return_attn:
            return output, attn_weights
        else:
            return output, None


class PositionwiseFeedForward(nn.Module):
    """
    前馈全连接网络（Position-wise FFN）
    对序列中每个位置独立计算，作用：引入非线性
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    复制 N 份相同的层
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    """
    Encoder 层：
    自注意力 + 残差 + LayerNorm + 前馈网络
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.pre_norm:
            # Pre-Norm 方式（更稳定）
            x2 = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=src_mask)[0]
            x = x + self.dropout(x2)
            x2 = self.ff(self.norm2(x))
            x = x + self.dropout(x2)
            return x, None
        else:
            # Post-Norm 方式（论文原版）
            x2, attn = self.attn(x, x, x, mask=src_mask, return_attn=True)
            x = x + self.dropout(x2)
            x = self.norm1(x)
            x2 = self.ff(x)
            x = x + self.dropout(x2)
            x = self.norm2(x)
            return x, attn


class DecoderLayer(nn.Module):
    """
    Decoder 层：
    Masked Self-Attention + Cross-Attention + FFN
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        trg_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.pre_norm:
            # Pre-Norm Decoder
            x2 = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=trg_mask)[0]
            x = x + self.dropout(x2)
            x2 = self.cross_attn(self.norm2(x), enc_output, enc_output, mask=src_mask)[0]
            x = x + self.dropout(x2)
            x2 = self.ff(self.norm3(x))
            x = x + self.dropout(x2)
            return x, {}
        else:
            # Post-Norm Decoder
            x2, self_attn = self.self_attn(x, x, x, mask=trg_mask, return_attn=True)
            x = x + self.dropout(x2)
            x = self.norm1(x)
            x2, cross_attn = self.cross_attn(x, enc_output, enc_output, mask=src_mask, return_attn=True)
            x = x + self.dropout(x2)
            x = self.norm2(x)
            x2 = self.ff(x)
            x = x + self.dropout(x2)
            x = self.norm3(x)
            return x, {"self_attn": self_attn, "cross_attn": cross_attn}


class Encoder(nn.Module):
    """
    堆叠多个 EncoderLayer
    """
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        attn_list = []
        x = src
        for layer in self.layers:
            x, attn = layer(x, src_mask=src_mask)
            attn_list.append(attn)
        return x, attn_list


class Decoder(nn.Module):
    """
    堆叠多个 DecoderLayer
    """
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)

    def forward(self, trg: torch.Tensor, enc_output: torch.Tensor, trg_mask: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        attn_store = {"self": [], "cross": []}
        x = trg
        for layer in self.layers:
            x, attn = layer(x, enc_output, trg_mask=trg_mask, src_mask=src_mask)
            if isinstance(attn, dict):
                attn_store["self"].append(attn.get("self_attn"))
                attn_store["cross"].append(attn.get("cross_attn"))
        return x, attn_store


class Transformer(nn.Module):
    """
    Transformer 总体架构：
    - Embedding + 位置编码
    - Encoder × N
    - Decoder × N
    - 输出层（映射到词表大小）
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        num_layers: int = 3,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # 输入 embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # 构造 Encoder / Decoder 堆叠
        enc_layer = EncoderLayer(d_model, n_heads, d_ff, dropout, pre_norm=pre_norm)
        dec_layer = DecoderLayer(d_model, n_heads, d_ff, dropout, pre_norm=pre_norm)
        self.encoder = Encoder(enc_layer, num_layers)
        self.decoder = Decoder(dec_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch, src_len)
        device = src.device
        # True 表示允许（非 PAD），False 表示屏蔽（PAD）
        src_pad_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,src_len)
        return src_pad_mask

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        # trg: (batch, trg_len)
        device = trg.device
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,trg_len)
        trg_len = trg.shape[1]
        # 下三角矩阵，防止看到未来 (trg_len, trg_len)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
        trg_sub_mask = trg_sub_mask.unsqueeze(0).unsqueeze(1)  # (1,1,trg_len,trg_len)
        trg_mask = trg_pad_mask & trg_sub_mask  # broadcast to (batch,1,trg_len,trg_len)
        return trg_mask

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        返回:
        - logits: (batch, trg_len, vocab_size)
        - attention 权重（便于可视化/调试）
        """
        # Embedding + PosEncoding
        src_emb = self.pos_enc(self.tok_embed(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_enc(self.tok_embed(trg) * math.sqrt(self.d_model))

        # mask
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # 编码
        enc_output, enc_attns = self.encoder(src_emb, src_mask=src_mask)
        # 解码
        dec_output, attn_store = self.decoder(trg_emb, enc_output, trg_mask=trg_mask, src_mask=src_mask)

        # 输出
        logits = self.fc_out(dec_output)
        return logits, {"encoder": enc_attns, "decoder": attn_store}

    # 下面两个方法是为了兼容训练脚本里直接调用 model.encode / model.decode
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        返回 encoder 的输出（embedding+pos + encoder stack）
        src: (batch, src_len)
        src_mask: optional mask (batch,1,1,src_len)
        """
        src_emb = self.pos_enc(self.tok_embed(src) * math.sqrt(self.d_model))
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        enc_output, _ = self.encoder(src_emb, src_mask=src_mask)
        return enc_output

    def decode(self, enc_output: torch.Tensor, ys: torch.Tensor, trg_mask: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        解码一步或多步，返回 logits (batch, seq_len, vocab)
        enc_output: encoder 输出 (batch, src_len, d_model)
        ys: 当前 decoder 输入 id 序列 (batch, cur_len)  — 包含起始符
        trg_mask / src_mask: 如果没有提供会自动根据 ys / enc_output 构造
        """
        # embed ys
        trg_emb = self.pos_enc(self.tok_embed(ys) * math.sqrt(self.d_model))
        if trg_mask is None:
            trg_mask = self.make_trg_mask(ys)
        # src_mask 需要提供原始 src 来计算时通常传入；若没有，尝试广播 True mask
        if src_mask is None:
            # create a permissive mask with same batch and src_len as enc_output
            bsz, src_len, _ = enc_output.size()
            src_mask = torch.ones((bsz, 1, 1, src_len), dtype=torch.bool, device=enc_output.device)
        dec_output, _ = self.decoder(trg_emb, enc_output, trg_mask=trg_mask, src_mask=src_mask)
        logits = self.fc_out(dec_output)
        return logits
