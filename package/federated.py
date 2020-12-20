"""Uses the estimator model_fn from estimator.py and uses it in a federated algorithm. The idea is to model the correct line by averaging the different clients' models. To simulate some difference, each client will have a different bias, and then if you average all the clients, you'll get the right line."""
import statistics

import tensorflow.compat.v1 as tf

from . import estimator


def global_seed(seed):
    tf.random.set_random_seed(seed)


def evaluate_model(model, sess):
    mse_values = []
    while True:
        try:
            _, mse = sess.run(model.eval_metric_ops)["mse"]
            mse_values.append(mse)
        except tf.errors.OutOfRangeError:
            break
    return statistics.mean(mse_values)


def model_fn_with_placeholders():
    features_placeholder = tf.placeholder(
        tf.float32, shape=[None, 1], name="features_placeholder"
    )
    labels_placeholder = tf.placeholder(
        tf.float32, shape=[None, 1], name="labels_placeholer"
    )

    model = estimator.model_fn(
        features_placeholder, labels_placeholder, tf.estimator.ModeKeys.TRAIN
    )

    return model


def copy_params(prefix_a, prefix_b):
    assignments = []

    for a, b in zip(
        tf.trainable_variables(scope=prefix_a),
        tf.trainable_variables(scope=prefix_b),
    ):
        assert a.name[len(prefix_a) :] == b.name[len(prefix_b) :]
        assignments.append(b.assign(a))

    return assignments


def add_params(prefix_a, prefix_b):
    assignments = []

    for a, b in zip(
        tf.trainable_variables(scope=prefix_a),
        tf.trainable_variables(scope=prefix_b),
    ):
        assert a.name[len(prefix_a) :] == b.name[len(prefix_b) :]
        assignments.append(b.assign_add(a))

    return assignments


def mean_and_assign_params(prefix_a, prefix_b, k):
    assignments = []

    for a, b in zip(
        tf.trainable_variables(scope=prefix_a),
        tf.trainable_variables(scope=prefix_b),
    ):
        assert a.name[len(prefix_a) :] == b.name[len(prefix_b) :]
        assignments.append(b.assign(a / k))

    return assignments


def zero(prefix):
    assignments = []

    for a in tf.trainable_variables(scope=prefix):
        assignments.append(a.assign(tf.zeros_like(a)))

    return assignments


def print_params(prefix):
    return [a for a in tf.trainable_variables(scope=prefix)]


def main():
    max_steps = 1
    num_clients = 1
    num_client_steps = 20
    server_prefix = "server"
    client_prefix = "client"
    sum_prefix = "sum"
    eval_prefix = "eval"

    global_seed(42)

    with tf.Session() as sess:
        with tf.variable_scope(server_prefix):
            server_model = model_fn_with_placeholders()

        with tf.variable_scope(sum_prefix):
            sum_model = model_fn_with_placeholders()
            # must be initialized to zero.

        with tf.variable_scope(client_prefix, reuse=tf.AUTO_REUSE):
            # somehow this needs to return different results based on a passed filename during sess.run
            client_features, client_labels = estimator.train_input_fn()
            client_model = estimator.model_fn(
                client_features, client_labels, tf.estimator.ModeKeys.TRAIN
            )

        with tf.variable_scope(eval_prefix):
            eval_features, eval_labels = estimator.eval_input_fn()
            eval_model = estimator.model_fn(
                eval_features, eval_labels, tf.estimator.ModeKeys.EVAL
            )

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for step in range(max_steps):
            # zero the sum because we are getting new client models
            sess.run(zero(sum_prefix))
            for k in range(num_clients):
                # copy wt to the client
                print("server before copying to client")
                print(sess.run(print_params(server_prefix)))
                sess.run(copy_params(server_prefix, client_prefix))
                print("server after copying to client")
                print(sess.run(print_params(server_prefix)))

                # train wt using data k to produce wk*
                for client_step in range(num_client_steps):
                    _, loss = sess.run(
                        [client_model.train_op, client_model.loss],
                    )
                    print(step, k, client_step, loss)

                # add wk*'s weights to sum(wk*)
                print("sum before getting client")
                print(sess.run(print_params(sum_prefix)))
                sess.run(add_params(client_prefix, sum_prefix))
                print("sum after getting client")
                print(sess.run(print_params(sum_prefix)))

            # wt+1 = mean(sum(w*))
            print("server before being assigned mean")
            print(sess.run(print_params(server_prefix)))
            sess.run(mean_and_assign_params(sum_prefix, server_prefix, num_clients))
            print("server after being assigned mean")
            print(sess.run(print_params(server_prefix)))

        sess.run(copy_params(server_prefix, eval_prefix))
        print(evaluate_model(eval_model, sess))


if __name__ == "__main__":
    main()
