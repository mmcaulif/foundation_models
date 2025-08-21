
import flax.linen as nn
from fm.nn.attention import MultiheadAttention
from fm.utils.positional import apply_sinusoidal_encoding

"""
TODO:
* Read original paper and implement anything else they mention e.g. dropout where 
appropriate, need to figure out how best to do this with Flax.linen: 
    https://colab.research.google.com/github/google/flax/blob/master/docs/guides/flax_sharp_bits.ipynb
* Create config default values and dummy training, can watch NanoGPT series for this
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
    @nn.compact
    def __call__(self, x):
        z = nn.LayerNorm(
            use_bias=False,
            use_scale=False,
            use_fast_variance=False
        )(x + self.module(x))
        return z


class TransformerBlock(nn.Module):
    emb_dim: int
    n_heads: int
    causal: bool = True
    @nn.compact
    def __call__(self, x):
        z = Sublayer(MultiheadAttention(self.emb_dim, self.n_heads, self.causal))(x)
        out = Sublayer(MLP(self.emb_dim))(z)
        return out


class Decoder(nn.Module):
    vocab_size: int
    embed_dim: int
    n_heads: int
    n_blocks: int
    @nn.compact
    def __call__(self, x):
        z = nn.Embed(self.vocab_size, self.embed_dim)(x)
        z = apply_sinusoidal_encoding(z)
        for _ in range(self.n_blocks):
            z = TransformerBlock(self.embed_dim, self.n_heads)(z)
        logits = nn.Dense(self.vocab_size)(z)
        return logits
