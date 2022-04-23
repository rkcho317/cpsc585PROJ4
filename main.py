from utils import decode_string, load_dataset
from model import create_model, Trainer
import tensorflow as tf
from tqdm import tqdm


def main():
    data = load_dataset()
    model = create_model()

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .map(lambda x: (x[:-1], x[1:]))\
        .shuffle(1000)\
        .batch(32, drop_remainder=True)

    trainer = Trainer(model, dataset, data)

    trainer.run(1000)

    # possibly slower method - Doesn't seem like it
    # model.compile(optimizer='adam', loss=loss_function)
    # model.fit(inputs, targets, epochs=10)

    # test predictions
    prediction = model(data[0:1])[0]
    prediction = tf.argmax(prediction, -1)
    print(decode_string(prediction))


if __name__ == '__main__':
    main()

