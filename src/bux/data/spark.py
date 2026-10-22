"""The pipeline on Spark: the map as mapInArrow, the reduce as applyInArrow.

    from pyspark.sql import SparkSession
    from bux.data import spark
    from bux.data.pipeline import Job, jobs

    session = SparkSession.builder.getOrCreate()
    for job in jobs(run.train):
        spark.prepare(job, session, "s3://bucket/packed")

A chunk's rows are the same as the local engine makes; which file holds them is up
to Spark. Needs pyspark 4, which bux does not depend on.
"""

import json

import pyarrow as pa
import pyarrow.fs
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from bux.data import documents, packing
from bux.data.pipeline import Job, shift
from bux.data.tokenizer import load


def _ddl(schema: pa.Schema) -> str:
    types = {pa.int64(): "bigint", pa.int32(): "int", pa.bool_(): "boolean"}
    return ", ".join(
        f"{f.name} array<{types[f.type.value_type]}>"
        if pa.types.is_list(f.type)
        else f"{f.name} {types[f.type]}"
        for f in schema
    )


def rows(job: Job, records: DataFrame) -> tuple[DataFrame, dict]:
    """Return the rows of `records` (a DataFrame with the source's column), and how
    many documents and tokens they hold."""
    docs = records.select(job.source.column).mapInArrow(job.map, _ddl(documents.SCHEMA))
    docs = docs.persist()
    counted = docs.agg(F.count("*"), F.sum(F.size("input_ids"))).first()
    n, tokens = (counted[0], counted[1] or 0) if counted else (0, 0)
    chunks = job.chunks(tokens)

    def reduce(key: tuple, table: pa.Table) -> pa.Table:
        batches, _ = job.reduce(key[0].as_py(), table)
        return pa.Table.from_batches(list(batches), packing.SCHEMA)

    chunk = F.shiftright("key", shift(chunks)).alias("chunk")
    out = docs.groupBy(chunk).applyInArrow(reduce, _ddl(packing.SCHEMA))
    return out, {"docs": n, "tokens": tokens}


def prepare(job: Job, session: SparkSession, cache: str) -> str:
    """Write `job`'s dataset to a folder under `cache` (any URI Spark and Arrow can
    write to), laid out as the local engine would."""
    folder = f"{cache.rstrip('/')}/{job.name}"
    out, stats = rows(job, session.read.parquet(job.source.path))
    out.write.parquet(folder)
    fs, path = pyarrow.fs.FileSystem.from_uri(folder)
    listed = fs.get_file_info(pyarrow.fs.FileSelector(path))
    parts = sorted(i.base_name for i in listed if i.base_name.endswith(".parquet"))
    meta = {"job": job.describe(), "files": [job.source.path], "parts": parts} | {
        "vocab_size": load(job.tokenizer).vocab_size,
        "stats": stats,
    }
    with fs.open_output_stream(f"{path}/dataset.json") as f:
        f.write(json.dumps(meta, indent=2).encode())
    return folder
