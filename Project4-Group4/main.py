from utils import decode_string, load_dataset
from model import GRUModel, Trainer, generate
import tensorflow as tf
from tqdm import tqdm


def main():
    data = load_dataset()
    model = GRUModel()

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .map(lambda x: (x[:-1], x[1:]))\
        .shuffle(1000)\
        .batch(32, drop_remainder=True)

    trainer = Trainer(model, dataset)

    print(generate(model))
    trainer.run(1000)

    # possibly slower method - Doesn't seem like it
    # model.compile(optimizer='adam', loss=loss_function)
    # model.fit(inputs, targets, epochs=10)



if __name__ == '__main__':
    main()

