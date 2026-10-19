import jax
import jax.numpy as jnp
import optax
from flax import nnx


class MLP(nnx.Module):
    def __init__(self, din: int, dhidden: int, dout: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear2(nnx.relu(self.linear1(x)))


@nnx.jit
def train_step(
    model: MLP, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array
) -> jax.Array:
    def loss_fn(model: MLP) -> jax.Array:
        return jnp.mean((model(x) - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def main() -> None:
    print(
        f"jax {jax.__version__} | backend: {jax.default_backend()} | devices: {jax.devices()}"
    )

    # Fit y = sin(x) on synthetic data.
    key = jax.random.key(0)
    x = jax.random.uniform(key, (256, 1), minval=-3.0, maxval=3.0)
    y = jnp.sin(x)

    model = MLP(1, 64, 1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    for step in range(501):
        loss = train_step(model, optimizer, x, y)
        if step % 100 == 0:
            print(f"step {step:4d}  loss {loss:.5f}")


if __name__ == "__main__":
    main()
