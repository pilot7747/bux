"""Training: what a .bux file says about how to train its model, and a trainer that does it.

``bux.train.config`` is plain data and needs no JAX, so the compiler and the editor
can import it. The rest trains a compiled ``bux.lang.ir.Run``, out of blocks that each
do one thing and can each be replaced:

    data       batches of rows, from the run's sources        a source per config type
    optim      learning-rate schedules and optimizers         one per config type
    objective  the loss, a weighted sum of named terms        a term is a function
    trainer    Trainer: compiles a step and runs the loop     override train_step
    hooks      what the loop does every so many steps         log, eval, checkpoint
    sinks      where what the hooks report goes               console, metrics.jsonl

``trainer.train(run)`` puts the run's blocks together. Config types find their
builders in registries (``bux.train.registry``), as layers do theirs.
"""
