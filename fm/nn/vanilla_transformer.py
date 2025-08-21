
import flax.linen as nn
from fm.nn.attention import MultiheadAttention
from fm.utils.positional import apply_sinusoidal_encoding

"""
TODO:
* Read original paper and implement anything else they mention e.g. dropout where 
appropriate, need to figure out how best to do this with Flax.linen: 
    https://colab.research.google.com/github/google/flax/blob/master/docs/guides/flax_sharp_bits.ipynb
* Create config default values and dummy training, can watch NanoGPT series for this
* Add default names and axis notation where possible!!!
"""


class MLP(nn.Module):
    emb_dim: int
    hidden_scale: int = 2
    @nn.compact
    def __call__(self, x):
        z = nn.Dense(self.emb_dim * self.hidden_scale)(x)
        z = nn.relu(z)
        out = nn.Dense(self.emb_dim)(z)
        return out


class Sublayer(nn.Module):
    module: nn.Module
    dropout: float = 0.1
    @nn.compact
    def __call__(self, x, training: bool = True):
        z = nn.Dropout(
            rate=self.dropout,
            deterministic=not training
        )(self.module(x))
        z = nn.LayerNorm(
            use_bias=False,
            use_scale=False,
            use_fast_variance=False
        )(x + z)
        return z


class TransformerBlock(nn.Module):
    emb_dim: int
    n_heads: int
    dropout: float = 0.1
    causal: bool = True
    @nn.compact
    def __call__(self, x, training: bool = True):
        z = Sublayer(
            module=MultiheadAttention(self.emb_dim, self.n_heads, self.causal),
            dropout=self.dropout
        )(x, training)
        out = Sublayer(
            module=MLP(self.emb_dim),
            dropout=self.dropout
        )(z)
        return out


class TransformerEmbed(nn.Module):
    vocab_size: int
    embed_dim: int
    dropout: float = 0.1
    @nn.compact
    def __call__(self, x, training: bool = False):
        z = nn.Embed(self.vocab_size, self.embed_dim)(x)
        z = apply_sinusoidal_encoding(z, self.embed_dim)
        out = nn.Dropout(rate=self.dropout, deterministic=not training)(z)
        return out


# TODO: make this more configurable so it can be used for Encoder and Decoders
class Decoder(nn.Module):
    vocab_size: int
    embed_dim: int
    n_heads: int
    n_blocks: int
    dropout: float = 0.1
    @nn.compact
    def __call__(self, x, training: bool = True):
        z = TransformerEmbed(self.vocab_size, self.embed_dim, self.dropout)(x, training)
        for _ in range(self.n_blocks):
            z = TransformerBlock(self.embed_dim, self.n_heads)(z, training)
        logits = nn.Dense(self.vocab_size)(z)
        return logits
