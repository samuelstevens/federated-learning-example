import tensorflow.compat.v1 as tf
import numpy as np


def make_client_dataset(name):
    if name == "c1":
        values = np.array([[4, 1, -7], [1, 1, -1], [5, 1, -9], [-4, 1, 9]], dtype=np.float32)
    elif name == "c2":
        values = np.array([[4, 1, -9], [1, 1, -3], [-4, 1, 7]], dtype=np.float32)
    else:
        raise ValueError(f"{name} is not a valid client name")

    def parse_row(row):
        features = row[:-1]
        label = row[-1:]
        return features, label

    dataset = tf.data.Dataset.from_tensor_slices(values)

    dataset = dataset.map(parse_row)

    dataset = dataset.shuffle(len(values)).repeat().batch(4)

    print(dataset)

    return dataset
