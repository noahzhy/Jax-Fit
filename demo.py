import os

import jax
import optax
from tqdm import tqdm
import jax.numpy as jnp
from flax.training import train_state
from flax.training import checkpoints
import tensorboardX as tbx
import tensorflow_datasets as tfds

from fit import *
from model import Model


def get_train_batches(batch_size=32):
    ds = tfds.load(name='mnist', split='train', as_supervised=True, shuffle_files=True)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)


def get_test_batches(batch_size=32):
    ds = tfds.load(name='mnist', split='test', as_supervised=True, shuffle_files=False)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)


def demo_train(key, epochs=10, batch_size=256):
    train_ds, test_ds = get_train_batches(batch_size), get_test_batches(batch_size)

    lr_fn = lr_schedule(2e-3, len(train_ds), epochs=epochs, warmup_epochs=2)

    model = Model()

    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, jnp.zeros((1, 28, 28, 1))),
        tx=optax.adam(lr_fn)
    )

    fit(state, train_ds, test_ds, epochs=epochs, log_name="mnist", lr_fn=lr_fn)


def demo_eval(key, ckpt_dir):
    model = Model()
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, jnp.zeros((1, 28, 28, 1))),
        tx=optax.adam(1e-3)
    )
    state = load_ckpt(state, ckpt_dir)
    test_ds = get_test_batches()
    accuracy = [state.test_step(batch) for batch in test_ds]
    accuracy = jnp.mean(jnp.array(accuracy))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    # demo_train(key)
    demo_eval(key, "checkpoints")
