import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# TODO: add BLEU score


@dataclass
class TrasnformerConfig:
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    n_heads: int = 8
    emb_size: int = 768
    hidden_size: int = 2048
    sequence_length: int = 512
    src_vocab_size: int = 30522
    trg_vocab_size: int = 30522
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    num_layers: int = 6
    n_heads: int = 8
    emb_size: int = 786
    sequence_length: int = 512
    vocab_size: int = 30522
    hidden_size: int = 2048
    dropout: float = 0.1


class CausalTransformer(nn.Module):
    def __init__(self, n_encoder_layers=6, n_decoder_layers=6, n_heads=8,
                 d_model=768, sequence_length=512, src_vocab_size=30522,
                 trg_vocab_size=30522, d_ff=2048, dropout=0.1):
        super(CausalTransformer, self).__init__()
        self.transformer = Transformer(
            n_encoder_layers, n_decoder_layers,
            n_heads, d_model, sequence_length,
            src_vocab_size, trg_vocab_size,
            dropout=dropout, d_ff=d_ff
        )

        self.l_norm = nn.LayerNorm([d_model])
        self.lm_head = nn.Linear(d_model, trg_vocab_size)

        self.apply(self._init_weights)

    def forward(self, src, trg, mask=None):
        output = self.transformer(src, trg, mask)
        return self.lm_head(self.l_norm(output))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Transformer(nn.Module):
    def __init__(self, n_encoder_layers=6, n_decoder_layers=6, n_heads=8,
                 d_model=768, sequence_length=512, src_vocab_size=30522,
                 trg_vocab_size=30522, pad_idx=0, dropout=0.1, d_ff=2048):
        super(Transformer, self).__init__()
        decoder_config = DecoderConfig(
            num_layers=n_decoder_layers, n_heads=8, emb_size=768, sequence_length=512, vocab_size=30522
        )
        self.encoder_embeddings = nn.Embedding(src_vocab_size, d_model, pad_idx)
        self.decoder_embeddings = nn.Embedding(trg_vocab_size, d_model, pad_idx)
        self.positional_embeddings = PositionalEmbedding(d_model, sequence_length, dropout)
        self.encoder = Encoder(n_encoder_layers, n_heads, d_model, sequence_length)
        self.decoder = Decoder(decoder_config)

        self.apply(self._init_weights)

    def forward(self, src, trg, mask=None):
        src = (self.encoder_embeddings(src))
        src = self.positional_embeddings(src)
        trg = self.decoder_embeddings(trg)
        trg = self.positional_embeddings(trg)
        encoder_output = self.encoder(src)
        output = self.decoder(encoder_output, trg, mask)

        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, sequence_length, dropout):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size

        positional_embedding = torch.empty((sequence_length, embedding_size))
        positions = torch.arange(0, sequence_length).unsqueeze(1)  # sequence_length, 1
        div_term = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10_000) / embedding_size)  # emb_size/2

        positional_embedding[:, ::2] = torch.sin(positions * div_term)  # sequence_length, emb_size/2
        positional_embedding[:, 1::2] = torch.cos(positions * div_term)
        # positional_embedding = positional_embedding.unsqueeze(0)  # 1, sequence_length, emb_size

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', positional_embedding)

    def forward(self, token_embedding: torch.Tensor):
        # batch, sequence_length, emb_size
        return self.dropout(
            token_embedding + self.pos_embedding[:token_embedding.size(1)]
        )


class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads, emb_size, sequence_length):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EncoderLayer(n_heads, emb_size, sequence_length))

    def forward(self, embedding: torch.Tensor):
        output = embedding
        for layer in self.layers:
            output = layer(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, emb_size, sequence_length, hidden_size=2048):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(n_heads, emb_size, sequence_length)
        self.norm1 = nn.LayerNorm([emb_size])
        self.dropout = nn.Dropout(0.1)
        # TODO: chnage to KANs
        self.ff = FeedForward(emb_size, hidden_size)
        self.norm2 = nn.LayerNorm([emb_size])

    def forward(self, x: torch.Tensor):
        attention = self.multihead_attention(x, x, x)
        norm_attention = self.norm1(x + self.dropout(attention))
        ff_output = self.ff(norm_attention)

        return self.norm2(ff_output + norm_attention)


class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.ff(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, emb_size, sequence_length):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.emb_size = emb_size
        self.sequence_length = sequence_length

        assert emb_size % n_heads == 0
        head_dim = emb_size // n_heads
        self.heads = nn.ModuleList(AttentionHead(emb_size, head_dim, sequence_length) for _ in range(n_heads))

        # TODO: KANs
        self.wo = nn.Linear(emb_size, emb_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        attention = torch.cat([head(q, k, v, mask) for head in self.heads], dim=-1)
        return self.wo(attention)


class AttentionHead(nn.Module):
    def __init__(self, emb_size, head_dim, sequence_length):
        super(AttentionHead, self).__init__()

        self.wq = nn.Linear(emb_size, head_dim, bias=False)
        self.wk = nn.Linear(emb_size, head_dim, bias=False)
        self.wv = nn.Linear(emb_size, head_dim, bias=False)

    def forward(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        return self.attention(q, k, v, mask)

    @staticmethod
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        k = k.transpose(-2, -1)
        _, _, d_k = q.size()  # batch, sequence_length, d_k
        attention_weights = q @ k / math.sqrt(d_k)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        normalized_attention = torch.softmax(attention_weights, dim=-1)
        return normalized_attention @ v


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(config.num_layers):
            self.layers.append(DecoderLayer(config))

    def forward(self, encoder_output, decoder_input, mask: torch.Tensor = None):
        hidden_state = decoder_input
        for layer in self.layers:
            hidden_state = layer(encoder_output, hidden_state, mask)

        return hidden_state


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.multihead_attn = MultiHeadAttention(config.n_heads, config.emb_size, config.sequence_length)
        self.cross_attn = MultiHeadAttention(config.n_heads, config.emb_size, config.sequence_length)
        self.layer_norm1 = nn.LayerNorm([config.emb_size])
        self.layer_norm2 = nn.LayerNorm([config.emb_size])
        self.layer_norm3 = nn.LayerNorm([config.emb_size])
        self.ff = FeedForward(config.emb_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.mask = torch.tril(torch.ones(config.sequence_length, config.sequence_length))
        self.register_buffer('tril_mask', self.mask)

    def forward(self, encoder_output, x, mask):
        seq_len = x.size(1)
        norm_x = self.layer_norm1(x)
        self_attn = self.multihead_attn(norm_x, norm_x, norm_x, self.tril_mask[:seq_len, :seq_len])
        x = self.dropout(self_attn) + x

        cross_attn = self.cross_attn(self.layer_norm2(x), encoder_output, encoder_output)
        x = self.dropout(cross_attn) + x

        out = self.dropout(self.ff(self.layer_norm3(x)))
        return out + x

