import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import *
from tqdm import tqdm
from time import sleep


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
        print(f"'{generate(self.model)}'")
        sleep(.01)

    @tf.function
    def _train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = loss_function(targets, prediction)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class GRUModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=len(alphabet) + 3, output_dim=64)
        self.gru_1 = layers.GRU(256, return_sequences=True)  # (batch_size, timesteps, 64)
        self.dense_1 = layers.Dense(len(alphabet) + 3)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.gru_1(x)
        x = self.dense_1(x)
        return x


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, PAD_TOKEN))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def generate(model):
    # not the best method, but works

    result = [START_TOKEN]

    for i in range(100):
        # prepare input
        src = tf.expand_dims(result, 0)

        # run model
        output = model(src)

        # look at last set of predictions - [0] => remove batch dim - [-1] => last entry
        probabilities = output[0][-1].numpy()

        # ignore start/pad tokens
        probabilities[START_TOKEN] = -99999
        probabilities[PAD_TOKEN] = -99999

        # prepare for sampling
        probabilities = tf.expand_dims(probabilities, 0)

        # sample
        selection = tf.random.categorical(probabilities, 1)

        # to int
        selection = tf.squeeze(selection).numpy()

        result.append(selection)

        if selection == END_TOKEN:
            break
    return decode_string(result)


if __name__ == '__main__':
    m = GRUModel()
    m.summary()
