import tensorflow.compat.v1 as tf
import numpy as np


def make_client_dataset(name):
    if name == "c1":
        values = np.array([[4, -7], [1, -1], [5, -9], [-4, 9]], dtype=np.float32)
    elif name == "c2":
        values = np.array([[4, -9], [1, -3], [-4, 7]], dtype=np.float32)
    elif name == "eval":
        values = np.array([[0, 0], [-10, 20], [5, -10]], dtype=np.float32)
    else:
        raise ValueError(f"{name} is not a valid client name")

    def parse_row(row):
        features = row[:-1]
        label = row[-1:]
        return features, label

    dataset = tf.data.Dataset.from_tensor_slices(values)

    dataset = dataset.map(parse_row)
    dataset = dataset.shuffle(len(values)).batch(4)

    if name != "eval":
        dataset = dataset.repeat()

    print(dataset)

    return dataset
