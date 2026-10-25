"""Data preparation: documents in Parquet files to rows of ``seq_len`` tokens, once.

A run trains on rows of ``seq_len`` tokens, each three arrays: ``input_ids``,
``position_ids`` (where each token is in its document) and ``loss_mask`` (whether it
is a target). Documents, raw text (``text(...)``) or conversations in the OpenAI
format (``chat(...)``), become rows before training, by a MapReduce job that any
MapReduce engine can run::

    Parquet ─map─▶ documents under random keys ─shuffle─▶ chunks ─reduce─▶ rows

- `bux.data.documents` is the map: it tokenizes records into documents, each
  under a key that hashes its content.
- `bux.data.packing` is the reduce: it packs a chunk of documents into rows.
- `bux.data.pipeline` defines the job (`~bux.data.pipeline.Job`), and runs it on
  this machine's cores (`~bux.data.pipeline.prepare`).
- `bux.data.spark` runs the same two functions on Spark.
- `bux.data.tokenizer` runs a Hugging Face tokenizer, in bulk.

``bux train`` and ``bux launch`` prepare what a run reads if it is missing, and
``bux prepare`` does it ahead. In Python, a run's jobs come from its settings::

    from bux.data.pipeline import jobs, prepare
    from bux.lang import compile_runs

    [run] = compile_runs(open("examples/experiments/speedrun.bux").read())
    for job in jobs(run.train):
        print(prepare(job))  # data/packed/text-0c8c782640, then the eval's

A prepared dataset is a folder named for a hash of what made it, under a cache
(``data/packed`` by default), which `bux.train.data` reads. :doc:`/data/index`
describes the settings and the design.
"""
