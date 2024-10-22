import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial


class Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=1, out_features=16, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=16, rngs=rngs)
        self.avg_pool1 = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

        self.conv2 = nnx.Conv(in_features=16, out_features=32, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.avg_pool2 = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

        self.conv3 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), rngs=rngs)
        self.bn3 = nnx.BatchNorm(num_features=64, rngs=rngs)

        self.dense1 = nnx.Linear(in_features=3136, out_features=256, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=256, out_features=10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool1(nnx.relu(self.bn1(self.conv1(x))))
        x = self.avg_pool2(nnx.relu(self.bn2(self.conv2(x))))
        x = nnx.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.dense1(x))
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    key = nnx.Rngs(0)
    model = Model(key)
    model.eval()
    x = jnp.ones((1, 28, 28, 1))
    y = model(x)
    nnx.display(model)