import jax
import jaxlib
import jax.numpy as jnp

print(jax.__version__)
print(jaxlib.__version__)

print(jax.devices())

print(jax.grad(jnp.linalg.det)(jnp.eye(2)))
