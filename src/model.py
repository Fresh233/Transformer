import torch
import math
from torch import nn
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class ModelArgs:
    n_embd: int
    n_heads: int
    dim: int
    dropout: float
    max_seq_len: int
    vocab_size: int
    block_size: int
    n_layer: int

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal=False):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask=None):
        bsz, seqlen, _ = q.shape
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        if self.is_causal:
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]

        if attention_mask is not None:
             scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_norm = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim * 4, args.dropout) # Common practice: FFN hidden is 4*dim

    def forward(self, x, src_mask):
        # 使用残差连接
        h = x + self.attention.forward(self.attention_norm(x), self.attention_norm(x), self.attention_norm(x), src_mask)
        out = h + self.feed_forward.forward(self.fnn_norm(h))

        # # 不使用残差连接
        # h = self.attention.forward(self.attention_norm(x), self.attention_norm(x), self.attention_norm(x), src_mask)
        # out = self.feed_forward.forward(self.fnn_norm(h))

        return out

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_1 = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim * 4, args.dropout) # Common practice: FFN hidden is 4*dim
        self.ffn_norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 使用残差连接
        x = x + self.mask_attention.forward(self.attention_norm_1(x), self.attention_norm_1(x), self.attention_norm_1(x), tgt_mask)
        h = x + self.attention.forward(self.attention_norm_2(x), enc_out, enc_out, src_mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        # # 不使用残差连接
        # x_after_mask_attn = self.mask_attention.forward(self.attention_norm_1(x), self.attention_norm_1(x),
        #                                                 self.attention_norm_1(x), tgt_mask)
        # h = self.attention.forward(self.attention_norm_2(x_after_mask_attn), enc_out, enc_out, src_mask)
        # out = self.feed_forward.forward(self.ffn_norm(h))

        return out

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)

class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(args.max_seq_len, args.n_embd)
        position = torch.arange(0, args.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.vocab_size is not None
        self.args = args
        self.src_tok_emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.tgt_tok_emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.pos_encoder = PositionalEncoding(args) # 位置编码
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        self.src_tok_emb.weight = self.lm_head.weight # Weight tying
        self.tgt_tok_emb.weight = self.lm_head.weight # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 使用位置编码
        src_emb = self.pos_encoder(self.src_tok_emb(src))
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt))

        # # 不使用位置编码
        # src_emb = self.src_tok_emb(src)
        # tgt_emb = self.tgt_tok_emb(tgt)

        enc_out = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, enc_out, src_mask, tgt_mask)
        logits = self.lm_head(output)
        return logits

    def encode(self, src, src_mask):
        # 使用位置编码
        return self.encoder(self.pos_encoder(self.src_tok_emb(src)), src_mask)
        # # 不使用位置编码
        # return self.encoder(self.src_tok_emb(src), src_mask)

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        # 使用位置编码
        return self.decoder(self.pos_encoder(self.tgt_tok_emb(tgt)), enc_out, src_mask, tgt_mask)
        # # 不使用位置编码
        # return self.decoder(self.tgt_tok_emb(tgt), enc_out, src_mask, tgt_mask)