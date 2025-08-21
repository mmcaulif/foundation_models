
import jax.numpy as jnp
import flax.linen as nn


"""
TODO:
* Add default names and axis notation where possible!!!
"""

def attention(q, k, v, d, causal):
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 3, 1))
    v = jnp.transpose(v, (0, 2, 1, 3))
    scores = (q @ k) / jnp.sqrt(d)
    if causal:
        mask = jnp.tril(scores)
        big_neg = jnp.finfo(jnp.float32).min
        scores = jnp.where(mask == 0, big_neg, scores)

    out = nn.softmax(scores, axis=-1) @ v
    out = out.transpose(0, 2, 1, 3)
    return out


class MultiheadAttention(nn.Module):
    emb_dim: int
    n_heads: int
    causal: bool = True
    @nn.compact
    def __call__(self, x):
        qkv_dim = self.emb_dim // self.n_heads
        q = nn.DenseGeneral(features=(self.n_heads, qkv_dim))(x)
        k = nn.DenseGeneral(features=(self.n_heads, qkv_dim))(x)
        v = nn.DenseGeneral(features=(self.n_heads, qkv_dim))(x)
        out = attention(q, k, v, qkv_dim, self.causal)
        return out.reshape(x.shape[:-1] + (self.emb_dim,))
    

class MultiQueryAttention(nn.Module):
    emb_dim: int
    n_heads: int
    causal: bool = True
    @nn.compact
    def __call__(self, x,):
        qkv_dim = self.emb_dim // self.n_heads
        q = nn.DenseGeneral(features=(self.n_heads, qkv_dim))(x)
        k = nn.DenseGeneral(features=(1, qkv_dim))(x).repeat(self.n_heads, axis=-2)
        v = nn.DenseGeneral(features=(1, qkv_dim))(x).repeat(self.n_heads, axis=-2)
        out = attention(q, k, v, qkv_dim, self.causal)
        return out.reshape(x.shape[:-1] + (self.emb_dim,))


class GroupedQueryAttention(nn.Module):
    emb_dim: int
    n_heads: int
    scale: int
    causal: bool = True
    @nn.compact
    def __call__(self, x):
        qkv_dim = self.emb_dim // self.n_heads
        q = nn.DenseGeneral(features=(self.n_heads, qkv_dim))(x)
        k = nn.DenseGeneral(features=(self.n_heads // self.scale, qkv_dim))(x).repeat(self.scale, axis=-2)
        v = nn.DenseGeneral(features=(self.n_heads // self.scale, qkv_dim))(x).repeat(self.scale, axis=-2)
        out = attention(q, k, v, qkv_dim, self.causal)
        return out.reshape(x.shape[:-1] + (self.emb_dim,))
