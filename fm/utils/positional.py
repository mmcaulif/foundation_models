import jax.numpy as jnp
import numpy as np


def apply_sinusoidal_encoding(x):
    d_model = x.shape[-1]
    seq_len = x.shape[-2]
    pos_encoding = np.zeros((seq_len, d_model))
    i = jnp.arange(0, d_model, 2)
    pos = jnp.expand_dims(jnp.arange(seq_len), -1)
    div_term = 1 / jnp.pow(10000, i/d_model)
    pos_encoding[:, 0::2] = jnp.sin(pos * div_term)
    pos_encoding[:, 1::2] = jnp.cos(pos * div_term)
    pos_encoding = jnp.expand_dims(pos_encoding, 0)
    return x + pos_encoding
