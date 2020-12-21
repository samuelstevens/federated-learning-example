# Federated Learning

This repo demonstrates a minimal example of federated learning in TensorFlow 1.15 using a model function that is compatible with the Estimator API.

Specifically:
- The `model_fn` is treated as a black box and does not use Keras or other high-level APIs
- The training algorithm is federated averaging, and is agnostic with regards to the client-side optimizer.
- There are only 3 copies of the model, not one for each client. 

## Getting Started

1. Install the dependencies with `pip install -r requirements.txt`. However, the `requirements.txt` file contains some extra dependencies. The bare minimum is installing `tensorflow==1.15`.
2. Run `python -m package.federated`
