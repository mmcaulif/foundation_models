
import flax.linen as nn
from fm.nn.attention import MultiheadAttention

"""
TODO:
* Dropout where appropriate
* Causal masking
* Sinusoidal positional embeddings
* Read original paper and implement anything they mention
* Create config default values and dummy training
"""


class LN(nn.Module):
    @nn.compact
    def __call__(self, x):
        out = nn.LayerNorm(
            use_bias=False,
            use_scale=False,
            use_fast_variance=False
        )(x)
        return out


class MLP(nn.Module):
    emb_dim: int
    upscale: int = 2
    @nn.compact
    def __call__(self, x):
        z = nn.Dense(self.emb_dim * 2)(x)
        z = nn.relu(z)
        out = nn.Dense(self.emb_dim)(z)
        return out


class TransformerBlock(nn.Module):
    emb_dim: int
    n_heads: int
    @nn.compact
    def __call__(self, x):
        z = LN()(x + MultiheadAttention(self.emb_dim, self.n_heads)(x))
        out = LN()(z + MLP(self.emb_dim)(z))
        return out
    

class Decoder(nn.Module):
    vocab_size: int
    embed_dim: int
    n_heads: int
    n_blocks: int
    @nn.compact
    def __call__(self, x):
        z = nn.Embed(self.vocab_size, self.embed_dim)(x)
        for _ in range(self.n_blocks):
            z = TransformerBlock(self.embed_dim, self.n_heads)(z)

        logits = nn.Dense(self.vocab_size)(z)
        return logits
