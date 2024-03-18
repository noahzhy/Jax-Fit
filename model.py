import jax
import jax.numpy as jnp
import flax.linen as nn


class Model(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3))
        self.dense1 = nn.Dense(features=256)
        self.dense2 = nn.Dense(features=10)

    @nn.compact
    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = jnp.mean(x, axis=(1, 2))
        x = nn.relu(self.dense1(x))
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 28, 28, 1))
    model = Model()
    tab = model.tabulate(key, x)
    print(tab)
