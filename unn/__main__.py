import logging

from . import mnist
from . import neuralnet


def main():
    logging.basicConfig(
        format="%(levelname)s %(message)s",
        level=logging.INFO
    )

    logging.info("Initializing empty neuralnet")

    net = neuralnet.Neuralnet((784, 30, 10))

    logging.info("Loading MNIST training dataset")

    training_data = mnist.load_dataset(
        "./data/train-images-idx3-ubyte.gz",
        "./data/train-labels-idx1-ubyte.gz"
    )

    logging.info("Initiating the training process")
    neuralnet.train(net, training_data, 30, 10, 3.0)


if __name__ == "__main__":
    main()
