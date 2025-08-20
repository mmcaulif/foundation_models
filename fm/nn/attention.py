
import jax.numpy as jnp
import flax.linen as nn


class MultiheadAttention(nn.Module):
    emb_dim: int
    n_heads: int
    @nn.compact
    def __call__(self, x):
        qkv_dim = self.emb_dim // self.n_heads
        q = nn.DenseGeneral(features=(qkv_dim, self.n_heads))(x)
        k = nn.DenseGeneral(features=(qkv_dim, self.n_heads))(x)
        v = nn.DenseGeneral(features=(qkv_dim, self.n_heads))(x)
        k_t = jnp.swapaxes(k, -1, -2)
        scores = (q @ k_t) / jnp.sqrt(qkv_dim)
        out = nn.softmax(scores, axis=-1) @ v
        return out.reshape(x.shape[:-1] + (self.emb_dim,))
    

class MultiQueryAttention(nn.Module):
    emb_dim: int
    n_heads: int
    @nn.compact
    def __call__(self, x):
        qkv_dim = self.emb_dim // self.n_heads
        q = nn.DenseGeneral(features=(qkv_dim, self.n_heads))(x)
        k = nn.DenseGeneral(features=(qkv_dim, 1))(x)
        k = jnp.repeat(k, self.n_heads, axis=-1)
        v = nn.DenseGeneral(features=(qkv_dim, 1))(x)
        v = jnp.repeat(v, self.n_heads, axis=-1)
        k_t = jnp.swapaxes(k, -1, -2)
        scores = (q @ k_t) / jnp.sqrt(qkv_dim)
        out = nn.softmax(scores, axis=-1) @ v
        return out.reshape(x.shape[:-1] + (self.emb_dim,))


class GroupedQueryAttention(nn.Module):
    emb_dim: int
    n_heads: int
    scale: int
    @nn.compact
    def __call__(self, x):
        qkv_dim = self.emb_dim // self.n_heads
        q = nn.DenseGeneral(features=(qkv_dim, self.n_heads))(x)
        k = nn.DenseGeneral(features=(qkv_dim, self.n_heads // self.scale))(x)
        k = jnp.repeat(k, self.scale, axis=-1)
        v = nn.DenseGeneral(features=(qkv_dim, self.n_heads // self.scale))(x)
        v = jnp.repeat(v, self.scale, axis=-1)
        k_t = jnp.swapaxes(k, -1, -2)
        scores = (q @ k_t) / jnp.sqrt(qkv_dim)
        out = nn.softmax(scores, axis=-1) @ v
        return out.reshape(x.shape[:-1] + (self.emb_dim,))
