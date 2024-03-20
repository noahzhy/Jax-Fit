import os

import jax
import optax
from tqdm import tqdm
import jax.numpy as jnp
from flax.training import train_state
from flax.training import checkpoints
import tensorboardX as tbx


def lr_schedule(base_lr, steps_per_epoch, epochs=100, warmup_epochs=5):
    return optax.warmup_cosine_decay_schedule(
        init_value=base_lr / 10,
        peak_value=base_lr,
        warmup_steps=steps_per_epoch * warmup_epochs,
        decay_steps=steps_per_epoch * (epochs - warmup_epochs),
    )


class TrainState(train_state.TrainState):
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params, batch):
            x, y = batch
            logits = state.apply_fn(params, x)
            loss = optax.softmax_cross_entropy(
                logits=logits,
                labels=jax.nn.one_hot(y, num_classes=10),
            ).mean()
            return loss, {"cross_entropy": loss}

        (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss, loss_dict

    @jax.jit
    def test_step(state, batch):
        x, y = batch
        logits = state.apply_fn(state.params, x)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return accuracy


# load the model
def load_ckpt(state, ckpt_dir, prefix="checkpoint_", step=None):
    return checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=state,
        step=step,
        prefix=prefix,
    )


def fit(state, train_ds, test_ds, epochs=100, log_name="default", lr_fn=None, val_frequency=1):
    tbx_writer: object = tbx.SummaryWriter("logs/{}".format(log_name))
    best = -1
    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_ds)
        for batch in pbar:
            state, loss, _dict = state.train_step(batch)
            lr = lr_fn(state.step)

            for key, value in _dict.items():
                tbx_writer.add_scalar(key, value, state.step)

            tbx_writer.add_scalar("loss", loss, state.step)
            tbx_writer.add_scalar("learning rate", lr, state.step)
            pbar.set_description(f"epoch: {epoch:3d}, loss: {loss:.4f}, lr: {lr:.4f}")

        if epoch % val_frequency == 0:
            accuracy = jnp.array([])
            for batch in test_ds:
                accuracy = jnp.append(accuracy, state.test_step(batch))
            accuracy = accuracy.mean()
            tbx_writer.add_scalar("accuracy", accuracy, epoch)
            print(f"epoch: {epoch:3d}, accuracy: {accuracy:.4f}")

            if accuracy > best:
                best = accuracy
                parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
                if not os.path.exists(parent_dir): os.makedirs(parent_dir)
                checkpoints.save_checkpoint(
                    ckpt_dir=parent_dir,
                    target=state,
                    step=epoch,
                    overwrite=True,
                    keep=1,)

    tbx_writer.close()
