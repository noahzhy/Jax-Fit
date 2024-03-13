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
    lr_fn: object
    loss_fn: object
    eval_fn: object
    log_name: str = "model"
    # to save checkpoints
    val_frequency: int = 5

    def train_step(self, batch):
        (loss, loss_dict), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(self.params, batch, self.apply_fn)
        self = self.apply_gradients(grads=grads)
        return self, loss, loss_dict

    def fit(self, train_ds, test_ds, epochs=100):
        tbx_writer: object = tbx.SummaryWriter("logs/{}".format(self.log_name))
        best = 0.0
        for epoch in range(1, epochs + 1):
            pbar = tqdm(train_ds)
            for batch in pbar:
                self, loss, _dict = self.train_step(batch)
                lr = self.lr_fn(self.step)

                for key, value in _dict.items():
                    tbx_writer.add_scalar(key, value, self.step)

                tbx_writer.add_scalar("loss", loss, self.step)
                tbx_writer.add_scalar("learning rate", lr, self.step)
                pbar.set_description(f"epoch: {epoch:3d}, loss: {loss:.4f}, lr: {lr:.4f}")

            if epoch % self.val_frequency == 0:
                accuracy = jnp.array([])
                for batch in test_ds:
                    accuracy = jnp.append(accuracy, self.eval_fn(self.params, batch, self.apply_fn))
                accuracy = accuracy.mean()
                tbx_writer.add_scalar("accuracy", accuracy, self.step)
                print(f"epoch: {epoch:3d}, accuracy: {accuracy:.4f}")

                if accuracy > best:
                    best = accuracy
                    checkpoints.save_checkpoint(
                        ckpt_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints")),
                        target=self,
                        step=epoch,
                        overwrite=True,
                        keep=1,)

        tbx_writer.close()


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    epochs = 100
    batch_size = 256
    train_ds, test_ds = get_train_batches(batch_size), get_test_batches(batch_size)

    lr_fn = lr_schedule(2e-3, len(train_ds), epochs=epochs, warmup_epochs=5)

    model = Model()

    def loss_fn(params, batch, model):
        x, y = batch
        loss = optax.softmax_cross_entropy(
            jax.nn.log_softmax(model(params, x)),
            jax.nn.one_hot(y, 10)
        ).mean()
        return loss, {"cross_entropy": loss}

    def eval_fn(params, batch, model):
        x, y = batch
        logits = model(params, x)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return accuracy

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
