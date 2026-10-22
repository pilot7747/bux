"""Data preparation: tokenize documents in Parquet files (raw text, or conversations
in the OpenAI format) and pack them into rows of seq_len tokens, once, as a map and a
reduce that any MapReduce engine can run.

``bux.data.pipeline`` defines the job and runs it locally, ``bux.data.spark`` runs it
on Spark, and ``bux.train.data`` reads what they write.
"""
