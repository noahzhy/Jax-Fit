# Jax Fit

![Jax](https://img.shields.io/badge/Powered%20by-JAX-0f76ab.svg)

Get started easily with training a model using JAX, simply fit it.

## Installation

Copy the `fit.py` file to your project.

## Quick Start

```python
from fit import lr_schedule, TrainState

key = jax.random.PRNGKey(0)

train_ds, test_ds = your_dataset()

lr_fn = lr_schedule(base_lr=1e-3, steps_per_epoch=len(train_ds), epochs=100, warmup_epochs=5)

model = YourModel()

state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(key, jnp.ones(your_input_shape)),
    tx=optax.adam(lr_fn),)

state.fit(state, train_ds, test_ds, epochs=epochs, log_name="mnist", lr_fn=lr_fn)
```

## Usage

First, import the `fit` module in your training script.

```python
from fit import *
```

Then, only two things are required to consider: loss function and evaluation function. Define them as follows:

```python
def loss_fn(params, batch, model):
    x, y = batch
    loss = optax.softmax_cross_entropy(
        jax.nn.log_softmax(model(params, x)),
        jax.nn.one_hot(y, 10)
    ).mean()
    return loss, {"cross_entropy": loss}
```

Notice that your loss function should return a loss value and a dictionary which you want to log to tensorboard.

```python
def eval_fn(params, batch, model):
    x, y = batch
    logits = model(params, x)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return accuracy
```

Prepare your dataset and data loaders, pytorch data loaders, tensorflow datasets are both supported.

tfds:

```python
ds = tfds.load("mnist", split="train", as_supervised=True)
train_ds = ds.take(50000).map(lambda x, y: (x / 255, y))
```

torch:

```python
ds = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_ds = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
```

By the way, `lr_schedule` is used to create the learning rate function, which is required by the `TrainState` object. You can define your own learning rate function, or use the default one:

```python
lr_fn = lr_schedule(base_lr=1e-3, steps_per_epoch=len(train_ds), epochs=100, warmup_epochs=5):
```

Now, create and fit your model:

```python
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(key, jnp.ones((1, 28, 28, 1))),
    tx=optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_fn)),
    lr_fn=lr_fn,
    eval_fn=eval_fn,
    loss_fn=loss_fn,)

state.fit(train_ds, test_ds, epochs=epochs)
```

## Summary

You can open the tensorboard to see the training process or check any loss and accuracy metrics.
