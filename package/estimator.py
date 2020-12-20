import numpy as np
import tensorflow.compat.v1 as tf


def model_fn(features, labels, mode, params=None, initializer=None):
    del params

    linear_model = tf.layers.Dense(units=1)

    y_pred = linear_model(features)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=y_pred)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            "mse": tf.metrics.mean_squared_error(
                labels=labels, predictions=y_pred, name="mse_metric"
            )
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def build_input_fn(features, labels, batch_size=2, finite=False):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if not finite:
        dataset = dataset.repeat()

    dataset = dataset.shuffle(1000).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def train_input_fn():
    features = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float32).T
    labels = np.array(
        [[0.0, -2, -4, -6, -8, -10, -12, -14, -16, -18]], dtype=np.float32
    ).T

    return build_input_fn(features, labels)


def eval_input_fn():
    features = np.array([[11.0, 12.0, 13.0, 14.5]], dtype=np.float32).T
    labels = np.array([[-20.0, -22.0, -24, -27]], dtype=np.float32).T

    return build_input_fn(features, labels, 1, finite=True)


def main():
    estimator = tf.estimator.Estimator(model_fn=model_fn)

    estimator.train(input_fn=train_input_fn, max_steps=1000)

    metrics = estimator.evaluate(input_fn=eval_input_fn)

    print(metrics)


if __name__ == "__main__":
    main()
