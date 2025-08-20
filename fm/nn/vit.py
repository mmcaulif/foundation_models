
import jax
import jax.numpy as jnp
import flax.linen as nn

from fm.nn.vanilla_transformer import TransformerBlock


class VIT(nn.Module):
    n_classes: int
    embed_dim: int
    n_heads: int
    n_blocks: int
    @nn.compact
    def __call__(self, x):
        z = nn.Conv()(x)
        z = jnp.reshape(z)

        classification_token = self.param(
            'classification_token',
            nn.initializers.lecun_normal(),
            self.embed_dim,
            jnp.float32,
        )
        
        exit()

        for _ in range(self.n_blocks):
            z = TransformerBlock(self.embed_dim, self.n_heads)(z)

        z_classification = z[...,0]
        logits = nn.Dense(self.n_classes)(z_classification)
        return logits


# class LinearTransform(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         z = jax.lax.conv_general_dilated(x, filter_shape=(4,4), window_strides=(3,3), padding="VALID")
#         return z
#         # return out



# x = jnp.ones((1, 1, 16, 16))

# model = LinearTransform()
# _model = nn.Conv
# key = jax.random.PRNGKey(0)
# params = model.init(key, x)
# y = model.apply(params, x)
# print(y.shape)

# P = 4 # Patch size
# img_size = 8

# square = jnp.arange(jnp.power(img_size, 2)).reshape(1, 1, img_size, img_size)
# print(square)

# square_unf = jax.lax.conv_general_dilated_patches(
#     lhs=square,
#     filter_shape=(P,P),
#     window_strides=(P,P),
#     padding='VALID'
# )



# square_unf = square_unf.transpose(0, 2, 3, 1)
# print(square_unf.shape)
# square_unf = square_unf.reshape(square_unf.shape[0], (square_unf.shape[1]*square_unf.shape[2]), -1)

# print(square_unf.shape)
# print(square_unf)
# exit()


