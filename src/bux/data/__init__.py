"""Data preparation: documents (raw text, or conversations in the OpenAI format) in
Parquet files, tokenized and packed into rows of seq_len tokens, once, as a map and a
reduce that any MapReduce engine can run.

``bux.data.pipeline`` holds the job and the engine on this machine,
``bux.data.spark`` the one on Spark; ``bux.train.data`` reads what they write.
"""
