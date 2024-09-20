# Jax Fit

![Jax](https://img.shields.io/badge/Powered%20by-JAX-0f76ab.svg) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Before you start, you should have some experience in PyTorch or TensorFlow. The Jax may be a difficult library to learn, but it's worth it.

Get started easily with training a model using JAX, simply fit it. If you are familiar with PyTorch Lightning or TensorFlow Keras, you will love this library. It's a simple and lightweight library for training your model with JAX in a few lines of code.

[TOC]

## Requirements
* **jax** (jax, jaxlib)
* **flax** to define your model
* **optax** for optimizer, learning rate schedule, and loss function
* **orbax** to save the checkpoints
* **tqdm** to show the progress bar
* **tensorboardX** to log the training process

## Installation

Fork this repository or copy the `fit.py` file to your project.

## Quick Start

It's a template for training your model. But only three key parts of code are required to modify in your script.

```python
import jax, flax, optax, orbax
from fit import lr_schedule, TrainState

# prepare your dataset
train_ds, test_ds = your_dataset()
# lr schedule
lr_fn = lr_schedule(
    base_lr=1e-3,
    steps_per_epoch=len(train_ds),
    epochs=100,
    warmup_epochs=5,
)

# key 1: your model
model = YourModel()

# init key and model
key = jax.random.PRNGKey(0)
x = jnp.ones((1, 28, 28, 1)) # MNIST example input
var = model.init(key, x, train=True)

state = TrainState.create(
    apply_fn=model.apply,
    params=var['params'],
    batch_stats=var['batch_stats'],
    tx=optax.inject_hyperparams(optax.adam)(lr_fn),
)

# your training step, the template in the next section
@jax.jit
def loss_fn():
    # key 2: your loss function
    ...
    return state, loss_dict, opt_state

# your evaluation step
@jax.jit
def eval_step():
    # key 3: your evaluation function
    ...
    return acc

fit(state, train_ds, test_ds,
    loss_fn=loss_fn,
    eval_step=eval_step,
    eval_freq=1,
    num_epochs=10,
    log_name='mnist',
)
```

## Usage

Let's start with a simple example, training a model on the MNIST dataset. First, import the `fit` module in your training script.

```python
from fit import *
```

Before training, you need to define your model, loss function, and evaluation function. Let's start with the model.

### Model

The following is a very simple example of a model. The `setup` function is used to define the model structure, and the `__call__` function defines the forward pass of the model.

```python
class Model(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=16, kernel_size=(3, 3))
        self.dense1 = nn.Dense(features=10)

    # train=False for evaluation mode
    # if you use dropout or batch normalization
    # I bet you will use it
    @nn.compact
    def __call__(self, x, train=False):
        # simple conv + bn + relu + fully connected layer
        x = self.conv1(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        # dropout layer
        x = nn.Dropout(rate=0.5)(x, deterministic=not train)
        # flatten
        x = x.reshape((x.shape[0], -1))
        x = self.dense1(x)
        return x
```

Then, only two things are required to consider: loss function and evaluation function.

<!-- The `train_step` function is a template for training a model. The `state` object is the `TrainState` object, which contains the model parameters, optimizer state, and other necessary information. The `batch` object is the input data, and the `opt_state` object is the optimizer state. -->

<!-- Don't afraid of the complexity of the `train_step` function, it's just a template. You can copy and paste it to your script and modify the `loss_fn` function only. -->

### Loss Function is all you need

Let's focus on the `loss_fn` function. Let's start with the pseudo pytorch style code. It's helpful to understand the `loss_fn` function in Jax.

```python
def loss_fn():
    loss = criterion(logits, labels)
    return loss
```

Easy, right? Let's continue to let's keep.

```python
@jax.jit
def loss_fn(logits, labels):
    loss = optax.softmax_cross_entropy(
        logits,
        jax.nn.one_hot(labels, 10)
    ).mean()
    # put the losses you want to log to tensorboard
    loss_dict = {'loss': loss}
    return loss, loss_dict
```

Notice that your loss function should return a total loss value and a dictionary which you want to log to tensorboard.

### Evaluation Function

Now, let's move on to the evaluation function with the pseudo pytorch style code.

```python
def eval_step():
    true_x, true_y = data
    model.eval()
    pred_y = model(true_x)
    # your metric function such as accuracy in pytorch
    acc = metric(pred_y, true_y)
    return acc
```

In pytorch, you can use the `model.eval()` function to switch the model to evaluation mode. Because the dropout layer and batch normalization layer have different behaviors in training and evaluation mode. In Jax, you need to set the `train=False` argument in the `apply_fn` function. Notice that your model structure should be different in training and evaluation mode if you use the dropout layer or batch normalization layer, see the `__call__` function in the [Model](#model) section.

It's similar to the `train_step` function and only requires the `state` object and the `batch` object.

```python
@jax.jit
def eval_step(state: TrainState, batch):
    x, y = batch
    logits = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats,
        }, x, train=False)
    acc = jnp.equal(jnp.argmax(logits, -1), y).mean()
    return acc
```

## Data Preparation

Prepare your dataset and data loaders for training and evaluation. You can use the TensorFlow Datasets or Torchvision Datasets to load the any dataset you want. Here is an example of loading the MNIST dataset.

### TensorFlow Datasets

```python
ds = tfds.load("mnist", split="train", as_supervised=True)
train_ds = ds.take(50000).map(lambda x, y: (x / 255, y))
```

### Torchvision Datasets

```python
ds = torchvision.datasets.MNIST(
    root="data", train=True, download=True,
    transform=torchvision.transforms.ToTensor()
)
train_ds = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
```

## Learning Rate Schedule

By the way, `lr_schedule` is used to create the learning rate function, which is required by the `TrainState` object. You can define your own learning rate function, or use the default one:

```python
lr_fn = lr_schedule(base_lr=1e-3,
    steps_per_epoch=len(train_ds),
    epochs=100,
    warmup_epochs=5,
)
```

Furthermore, you can define your own chainable update transformations, check the `optax` library for more information.

```python
state = TrainState.create(
    apply_fn=model.apply,
    params=var['params'],
    batch_stats=var['batch_stats'],
    # chainable update transformations
    tx=optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_fn)),
)
```
    
Finally, call the `fit` function to start training.

```python
fit(state, train_ds, test_ds,
    loss_fn=loss_fn,
    eval_step=eval_step,
    # evaluate the model every N epochs (default 1)
    eval_freq=1,
    num_epochs=10,
    # log name for tensorboard
    log_name='mnist',
    # hyperparameters for the training process
    # such as batch size, learning rate, etc.
    # it's optional for you
    hparams={
        'batch_size': 32,
        'lr': 1e-3,
    },
)
```

## Visualization

You can open the tensorboard to see the training process or check any loss and accuracy metrics.

## Q & A

**What's the @jax.jit decorator?**

It's a decorator to compile the function to a single static function, which can be executed on GPU or TPU, if you want to speed up the training process especially for the your own loss function and evaluation function, you can add the `@jax.jit` decorator.

**What's the batch state and the dropout key?**

The batch state is used to store the batch normalization statistics, and the dropout key is used to generate the random mask for the dropout layer.
