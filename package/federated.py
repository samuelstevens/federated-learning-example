"""Uses the estimator model_fn from estimator.py and uses it in a federated algorithm. The idea is to model the correct line by averaging the different clients' models. To simulate some difference, each client will have a different bias, and then if you average all the clients, you'll get the right line."""


import statistics

import tensorflow.compat.v1 as tf

from . import estimator, data


def global_seed(seed):
    tf.random.set_random_seed(seed)


def evaluate_model(model, feed_dict, sess):
    mse_values = []
    print(sess.run(print_params("client")))
    while True:
        try:
            _, mse = sess.run(model.eval_metric_ops, feed_dict=feed_dict)["mse"]
            mse_values.append(mse)
        except tf.errors.OutOfRangeError:
            break
    return statistics.mean(mse_values)


def model_fn_with_placeholders(mode):
    features_placeholder = tf.placeholder(
        tf.float32, shape=[None, 1], name="features_placeholder"
    )

    labels_placeholder = tf.placeholder(
        tf.float32, shape=[None, 1], name="labels_placeholder"
    )

    model = estimator.model_fn(features_placeholder, labels_placeholder, mode)

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
    # region hyperparams

    max_steps = 10
    num_client_steps = 50
    server_prefix = "server"
    client_prefix = "client"
    sum_prefix = "sum"

    # endregion hyperparameters

    global_seed(42)
    client_names = ["c1", "c2"]
    num_clients = len(client_names)

    # client model weights
    with tf.variable_scope(client_prefix):
        # region training data

        # client iterators
        client_iters = []
        for name in client_names:
            dataset = data.make_client_dataset(name)
            iterator = dataset.make_one_shot_iterator()
            client_iters.append(iterator)

        iter_handle = tf.placeholder(tf.string, shape=[])  # [] means rank 0 => scalar
        train_data_iterator = tf.data.Iterator.from_string_handle(
            iter_handle, dataset.output_types, dataset.output_shapes
        )  # dataset refers to last dataset from the for loop above. This can be especially risky because it depends on this code being moved as a single block.
        features, labels = train_data_iterator.get_next()

        # endregion training data

        client_model = estimator.model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope('eval'):
        eval_model = estimator.model_fn(features, labels, tf.estimator.ModeKeys.EVAL)

    # region eval data
    eval_iter = data.make_client_dataset("eval").make_one_shot_iterator()
    # endregion eval data

    # copy of server weights
    with tf.variable_scope(server_prefix):
        model_fn_with_placeholders(tf.estimator.ModeKeys.TRAIN)

    # sum of client weights, before it's converted to an average
    with tf.variable_scope(sum_prefix):
        model_fn_with_placeholders(tf.estimator.ModeKeys.TRAIN)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        client_handles = []
        for iterator in client_iters:
            handle = sess.run(iterator.string_handle())
            client_handles.append(handle)

        eval_handle = sess.run(eval_iter.string_handle())

        # region main training loop

        for step in range(max_steps):
            # zero the sum because we are getting new client models
            sess.run(zero(sum_prefix))
            for k, client_handle in zip(range(num_clients), client_handles):
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
                        feed_dict={
                            iter_handle: client_handle,
                        },
                    )
                    print(sess.run(print_params(client_prefix)))
                    print(step, k, name, client_step, loss)

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

        # endregion
        sess.run(copy_params(server_prefix, 'eval'))
        print(
            evaluate_model(
                eval_model,
                {iter_handle: eval_handle},
                sess,
            )
        )


if __name__ == "__main__":
    main()
