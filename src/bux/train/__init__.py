"""Training: what a .bux file says about how to train its model, and a trainer that does it.

``bux.train.config`` is plain data and needs no JAX, so the compiler and the editor
can import it. ``bux.train.trainer`` runs a compiled ``bux.lang.ir.Run``.
"""
