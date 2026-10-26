"""Hyperparameter search: the trials of a run whose settings are distributions.

A setting written as a distribution is a dimension of a search::

    optimizer.lr = log_uniform(1e-4, 1e-2)
    optimizer.weight_decay = uniform(0, 0.2)
    objective = 0.7 * web_bpb + 0.3 * code_bpb
    search = search(method=tpe, trials=32, halving=halving(eta=3, rungs=3))

Each trial is a run of its own, at a point of the dimensions that the search's
method proposes (`bux.search.methods`): random, quasi-random (Sobol), TPE
(`bux.search.tpe`) or a Gaussian process (`bux.search.gp`). With `halving`, trials
train short runs first, and only the best of each rung train longer
(`bux.search.halving`). `bux launch` trains the trials side by side on the GPUs and
proposes the next as one ends; the search keeps them in its directory, so a search
launched again goes on where it stopped.
"""

from bux.search import methods

__all__ = ["methods"]
