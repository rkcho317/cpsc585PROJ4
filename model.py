import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import *
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.optimizer = tf.keras.optimizers.Adam()

    def run(self, num_epochs):
        for e in range(num_epochs):
            self.run_epoch(e)

    def run_epoch(self, epoch):
        with tqdm(self.dataset, f"Epoch {epoch}") as pbar:
            total_loss = 0
            count = 0
            for inputs, targets in pbar:
                loss = self._train_step(inputs, targets)
                total_loss += loss.numpy()
                count += 1
                pbar.set_postfix_str(f"Loss: {total_loss / count}")
        pbar.close()

    @tf.function
    def _train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = loss_function(targets, prediction)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


def create_model():
    result = keras.Sequential()
    result.add(layers.Embedding(input_dim=len(alphabet) + 3, output_dim=64))
    result.add(layers.GRU(64, return_sequences=True))  # (batch_size, timesteps, 64)
    result.add(layers.Dense(len(alphabet) + 3))
    return result


def loss_function(real, pred):
    loss_function.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, PAD_TOKEN))
    loss_ = loss_function.loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


if __name__ == '__main__':
    m = create_model()
    m.summary()
