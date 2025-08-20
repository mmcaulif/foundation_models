import jax
import jax.numpy as jnp
from fm.nn.vanilla_transformer import Decoder
from fm.nn.attention import GroupedQueryAttention, MultiQueryAttention


model = GroupedQueryAttention(512, 8, 2)

x = jnp.ones((10, 256))
key = jax.random.PRNGKey(0)
params = model.init(key, x)
y = model.apply(params, x)
print(y.shape)
exit()

model = Decoder(
    vocab_size=5_000,
    embed_dim=512,
    n_heads=8,
    n_blocks=6
)
key = jax.random.PRNGKey(0)
x = jnp.arange(0, 10*256).reshape((10, 256))
params = model.init(key, x)
n_params_per_layer = jax.tree.map(lambda x: jnp.prod(jnp.array(x.shape)), params)
n_params = jax.tree.reduce(jnp.add, n_params_per_layer)
print(n_params)
y = model.apply(params, x)
print(y.shape)
