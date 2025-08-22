import jax
import jax.numpy as jnp
import flax.linen as nn
from fm.nn.vanilla_transformer import Decoder
from fm.nn.attention import GroupedQueryAttention, MultiQueryAttention, MultiheadAttention


# model = GroupedQueryAttention(512, 8, 2)
# key = jax.random.PRNGKey(0)
# x = jax.random.normal(key, (1, 7, 64))
# params = model.init(key, x)
# y = model.apply(params, x)
# print(y.shape)
# exit()

model = Decoder(
    vocab_size=5_000,
    embed_dim=512,
    n_heads=8,
    n_blocks=6
)

key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)
dummy_x = jnp.ones((256, 64), dtype=jnp.int32)
params = model.init(key, dummy_x, training=False)
n_params_per_layer = jax.tree.map(lambda x: jnp.prod(jnp.array(x.shape)), params)
n_params = jax.tree.reduce(jnp.add, n_params_per_layer)
print(f"<<< Using Decoder architecture with {n_params} parameters >>>")

for _ in range(10):
    key, data_key, dropout_key = jax.random.split(key, 3)
    x = jax.random.randint(
        key=key,
        shape=(256, 32),
        minval=0,
        maxval=model.vocab_size
    )
    y = model.apply(params, x, rngs={"dropout": dropout_key})
    print(y.shape)
