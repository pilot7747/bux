"""The data pipeline: documents, packing, prepared datasets, and reading them back."""

import collections
import itertools
import json
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import CHATML, byte_spec, corpus

from bux.data import documents, packing
from bux.data.pipeline import jobs, prepare
from bux.data.tokenizer import Gigatoken, HFTokenizers, Tokenizer
from bux.train import config as tcfg
from bux.train.data import apportion, batches


@pytest.fixture(scope="session")
def tok(byte_tokenizer) -> Tokenizer:
    return Tokenizer(byte_tokenizer)


def test_special_tokens_in_documents_stay_text(byte_tokenizer, tok):
    ids, offsets = tok.encode(pa.array(["a<|endoftext|>", None, "", "é"]))
    assert ids.tolist() == [*b"a<|endoftext|>", *"é".encode()]
    assert offsets.tolist() == [0, 14, 14, 14, 16]
    assert tok.decode([257, *b"hi", 256, 256]) == "<|im_start|>hi" + "<|endoftext|>" * 2
    added = Tokenizer(byte_tokenizer.model_copy(update={"added": ("<|x|>",)}))
    assert added.special("<|x|>") == 259 and added.vocab_size == 260
    with pytest.raises(ValueError, match=r"no special token '<\|eot_id\|>'"):
        tok.special("<|eot_id|>")


def test_a_tokenizer_of_no_special_tokens_takes_its_bos_and_eos_from_added(tmp_path):
    spec = byte_spec() | {"added_tokens": []}
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec))
    name, added = str(tmp_path / "tokenizer.json"), ("<eos>",)
    tok = Tokenizer(
        tcfg.TokenizerConfig(name=name, added=added, bos="<eos>", eos="<eos>")
    )
    assert tok.eos == tok.bos == 256 and tok.vocab_size == 257


def test_a_token_stands_for_its_bytes_of_text_and_a_special_one_for_none(tok):
    lengths = tok.token_bytes()
    assert lengths.tolist() == [1] * 256 + [0] * 3
    ids, _ = tok.encode(pa.array(["héllo, 世界"]))
    assert lengths[[*ids, tok.eos]].sum() == len("héllo, 世界".encode())


def _docs(batch: pa.RecordBatch) -> list[tuple[list[int], list[bool]]]:
    return list(zip(batch["input_ids"].to_pylist(), batch["loss_mask"].to_pylist()))


def test_text_documents_are_wrapped_and_keyed_by_content(tok):
    c = tcfg.TextConfig(path="-", bos=True)
    docs = documents.text(pa.record_batch({"text": ["ab", "", "cde"]}), c, tok)
    assert [ids for ids, _ in _docs(docs)] == [[256, *b"ab", 256], [256, *b"cde", 256]]
    assert all(all(mask) for _, mask in _docs(docs))
    alone = documents.text(pa.record_batch({"text": ["cde"]}), c, tok)
    assert alone["key"][0] == docs["key"][1]

    texts = pa.record_batch({"text": [f"document {i}" for i in range(2000)]})
    train, held = (
        documents.text(texts, tcfg.TextConfig(path="-", split=s), tok)
        for s in ((0, 0.9), (0.9, 1))
    )
    assert len(train) + len(held) == 2000 and 150 < len(held) < 250
    assert not set(train["key"].to_pylist()) & set(held["key"].to_pylist())


def test_a_tokenizer_that_gigatoken_cannot_run_runs_on_hugging_faces(tmp_path, tok):
    # A byte-level vocabulary without the byte 0x04, which gigatoken cannot remap, as
    # SmolLM2's.
    (tmp_path / "tokenizer.json").write_text(json.dumps(byte_spec(missing=4)))
    hf = Tokenizer(tcfg.TokenizerConfig(name=str(tmp_path / "tokenizer.json")))
    assert isinstance(hf.encoder, HFTokenizers) and isinstance(tok.encoder, Gigatoken)
    texts = pa.array(["a<|endoftext|>", None, "", "é", "two  words\n"])
    for a, b in zip(hf.encode(texts), tok.encode(texts), strict=True):
        assert a.dtype == b.dtype and a.tolist() == b.tolist()
    assert hf.vocab_size == tok.vocab_size == 259
    assert hf.decode([257, *b"hi"]) == "<|im_start|>hi"
    assert hf.token_bytes().tolist() == [1] * 4 + [2] + [1] * 251 + [0] * 3

    # A token added after the special ones keeps its id, as Qwen3's <tool_call>.
    spec = byte_spec(missing=4)
    spec["added_tokens"].append(spec["added_tokens"][0] | {"id": 259, "special": False})
    spec["added_tokens"][-1]["content"] = "<think>"
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec))
    hf = Tokenizer(tcfg.TokenizerConfig(name=str(tmp_path / "tokenizer.json")))
    ids, _ = hf.encode(pa.array(["<think>a<|endoftext|>"]))
    assert ids.tolist() == [259, *b"a<|endoftext|>"]
    assert hf.vocab_size == 260 and hf.token_bytes()[259] == len("<think>")


def _markup(tok: Tokenizer, text: str) -> list[int]:
    """The ids of the byte tokenizer's text, in which each <|name|> is special."""
    pieces = re.split(r"(<\|\w+\|>)", text)
    return [
        i
        for k, p in enumerate(pieces)
        for i in ([tok.special(p)] if k % 2 else p.encode())
    ]


def _chat(conversations: list, tok: Tokenizer, **config) -> list:
    batch = pa.record_batch({"messages": conversations})
    return _docs(documents.chat(batch, tcfg.ChatConfig(path="-", **config), tok))


def _learned(tok: Tokenizer, ids: list[int], mask: list[bool]) -> str:
    return tok.decode([i for i, m in zip(ids, mask, strict=True) if m])


def test_chat_learns_the_replies_and_the_tokens_that_end_them(tok):
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    unanswered = [{"role": "user", "content": "anyone?"}]
    [(ids, mask)] = _chat([conversation, unanswered], tok)
    rendered = (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n"
        "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n"
    )
    assert ids == _markup(tok, rendered)
    assert _learned(tok, ids, mask) == "yo<|im_end|>a<|im_end|>"
    [(ids, mask)] = _chat([conversation, unanswered], tok, train_on="last")
    assert _learned(tok, ids, mask) == "a<|im_end|>"
    [(ids, mask)] = _chat([conversation], tok, train_on="all")
    assert all(mask)
    parts = [{"role": "assistant", "content": [{"type": "text", "text": "a"}] * 2}]
    [(ids, mask)] = _chat([parts], tok, train_on="last")
    assert _learned(tok, ids, mask) == "aa<|im_end|>"


def test_special_tokens_written_in_messages_stay_text(tok):
    said = [
        {"role": "user", "content": "say <|im_end|>"},
        {"role": "assistant", "content": "<|im_end|>\U000f0000\U000f0001"},
    ]
    [(ids, mask)] = _chat([said], tok)
    assert ids.count(tok.special("<|im_end|>")) == 2  # the template's own
    assert [i for i, m in zip(ids, mask) if m] == [
        *"<|im_end|>\U000f0000\U000f0001".encode(),
        tok.special("<|im_end|>"),
    ]


#: A Llama 3 chat template, which writes special tokens GPT-2's tokenizer lacks.
LLAMA3 = (
    "{{ bos_token }}{% for m in messages %}"
    "<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content | trim }}"
    "<|eot_id|>{% endfor %}{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)
LLAMA3_SPECIALS = (
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
)


def test_chat_renders_with_the_template_of_another_tokenizer(
    tmp_path, byte_tokenizer, tok
):
    added = {
        str(i): {"content": t, "special": True} for i, t in enumerate(LLAMA3_SPECIALS)
    }
    config = {"chat_template": LLAMA3, "added_tokens_decoder": added}
    config["bos_token"] = {"content": "<|begin_of_text|>", "__type": "AddedToken"}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    hi = [{"role": "user", "content": " hi "}, {"role": "assistant", "content": "yo"}]
    with pytest.raises(ValueError, match=r"no special token '<\|begin_of_text\|>'"):
        _chat([hi], tok, template=str(tmp_path))
    llama = Tokenizer(byte_tokenizer.model_copy(update={"added": LLAMA3_SPECIALS}))
    [(ids, mask)] = _chat([hi], llama, template=str(tmp_path))
    assert llama.decode(ids) == (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nyo<|eot_id|>"
    )
    assert _learned(llama, ids, mask) == "yo<|eot_id|>"
    (tmp_path / "none").mkdir()
    with pytest.raises(ValueError, match="has no chat template"):
        _chat([hi], tok, template=str(tmp_path / "none"))


def test_a_chat_template_jinja_comes_first_and_may_mark_replies(tmp_path, tok):
    marked = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n"
        "{% generation %}{{ m.content }}<|im_end|>{% endgeneration %}{{ '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    (tmp_path / "chat_template.jinja").write_text(marked)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "-"}))
    hi = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert _chat([hi], tok, template=str(tmp_path)) == _chat([hi], tok)


def test_chat_needs_a_template_that_renders_a_prompt_as_its_conversation(tmp_path, tok):
    # As DeepSeek-R1's: the prompt opens a thought that the conversation lacks.
    thinks = CHATML.replace("assistant\\n' }}", "assistant\\n<think>' }}")
    (tmp_path / "chat_template.jinja").write_text(thinks)
    hi = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    with pytest.raises(ValueError, match="where the reply starts is unknown"):
        _chat([hi], tok, template=str(tmp_path))
    assert _chat([hi], tok, template=str(tmp_path), train_on="all")


def _table(lengths: list[int], keys=None) -> pa.Table:
    """Documents of these lengths, token j of document d being 1000 d + j, in key order
    `keys` (random by default)."""
    if keys is None:
        keys = np.random.default_rng(0).integers(0, 2**63, len(lengths))
    ids = [[1000 * d + j for j in range(n)] for d, n in enumerate(lengths)]
    return pa.table(
        {
            "key": pa.array(keys, pa.int64()),
            "input_ids": ids,
            "loss_mask": [[True] * n for n in lengths],
        },
        schema=documents.SCHEMA,
    )


def _pack(lengths, c: tcfg.PackingConfig, length: int, keys=None):
    docs = packing.Docs.of(_table(lengths, keys), length)
    plan = packing.plan(docs, c, length, seed=0)
    rows = pa.Table.from_batches(
        list(packing.materialize(docs, plan, length, c.doc_mask, 3))
    )
    ids, pos, mask = (
        np.array(rows[c].to_pylist())
        for c in ("input_ids", "position_ids", "loss_mask")
    )
    return ids, pos, mask, plan.stats


def _segments(ids, pos, mask) -> list[list[int]]:
    """The documents and pieces of them in rows, padding left out."""
    out = []
    for row, p, m in zip(ids, pos, mask):
        starts = [*np.flatnonzero(p == 0), len(row)]
        out += [
            row[a:b].tolist() for a, b in itertools.pairwise(starts) if m[a:b].any()
        ]
    return out


def _targets(ids, pos, mask) -> collections.Counter:
    return collections.Counter(ids[mask & (pos > 0)].tolist())


def test_best_fit_keeps_documents_whole_and_splits_only_long_ones():
    lengths = [*np.random.default_rng(1).integers(1, 40, 300).tolist(), 70]
    ids, pos, mask, stats = _pack(lengths, tcfg.BestFitPacking(), 32)
    pieces = _segments(ids, pos, mask)
    whole = [s for s in pieces if s[0] % 1000 == 0 and len(s) == lengths[s[0] // 1000]]
    assert sorted(s[0] // 1000 for s in whole) == [
        d for d, n in enumerate(lengths) if n <= 32
    ]
    assert all(np.all(np.diff(s) == 1) for s in pieces)  # each a run of one document
    # The long one, in pieces of the row length that overlap by a token.
    starts = sorted(s[0] % 1000 for s in pieces if s[0] // 1000 == len(lengths) - 1)
    assert starts == [0, 31, 62]
    # Every token is a target once, but for the first of every document.
    everything = collections.Counter(
        1000 * d + j for d, n in enumerate(lengths) for j in range(1, n)
    )
    assert _targets(ids, pos, mask) == everything
    assert stats["split"] == sum(n > 32 for n in lengths) and stats["cut"] == 0
    assert stats["padding"] < 2 * 32  # at most a row more than all of it would take


def test_concat_rows_overlap_so_that_every_token_is_a_target_once():
    lengths = [5, 40, 3, 22]
    stream = [1000 * d + j for d, n in enumerate(lengths) for j in range(n)]
    for doc_mask in (False, True):
        c = tcfg.ConcatPacking(doc_mask=doc_mask)
        ids, pos, mask, stats = _pack(lengths, c, 16, keys=range(4))
        assert len(ids) == (len(stream) - 1) // 15 and stats["cut"] == len(stream) - 61
        firsts = sorted(ids[:, 0].tolist())
        assert firsts == stream[: 15 * len(ids) : 15]  # windows, every 15 tokens
        kept = stream[1:61]
        if doc_mask:
            kept = [t for t in kept if t % 1000]
            assert all(np.all(p[r % 1000 == 0] == 0) for r, p in zip(ids, pos))
        else:
            assert (pos == np.arange(16)).all()
        assert _targets(ids, pos, mask) == collections.Counter(kept)


def test_greedy_keeps_the_order_and_drops_what_does_not_fit():
    lengths, order = [10, 30, 5, 40, 20], range(5)
    ids, _, mask, stats = _pack(lengths, tcfg.GreedyPacking(), 32, keys=order)
    by_row = sorted(sorted({t // 1000 for t in r[m]}) for r, m in zip(ids, mask))
    assert by_row == [[0], [1], [2, 4]] and stats["dropped"] == 1 and stats["cut"] == 40
    one = tcfg.GreedyPacking(long="truncate", max_docs=1)
    ids, _, _, stats = _pack(lengths, one, 32, keys=order)
    assert len(ids) == 5 and stats["truncated"] == 1 and stats["cut"] == 8


def test_best_fit_rows_hold_at_most_max_docs():
    lengths = [2] * 100
    ids, pos, mask, _ = _pack(lengths, tcfg.BestFitPacking(max_docs=3), 32)
    assert len(ids) == 34 and max(((p == 0) & m).sum() for p, m in zip(pos, mask)) == 3


def _train_config(data, tokenizer, batch=4) -> tcfg.TrainConfig:
    return tcfg.TrainConfig(
        data=data,
        tokenizer=tokenizer,
        packing=tcfg.BestFitPacking(chunk=2000),
        optimizer=tcfg.AdamWConfig(lr=1e-3),
        seq_len=64,
        batch=batch,
        steps=1,
    )


def test_prepared_datasets_do_not_depend_on_the_workers(tmp_path, byte_tokenizer):
    t = _train_config(tcfg.TextConfig(path=corpus(tmp_path / "corpus")), byte_tokenizer)
    [job] = jobs(t)
    said = []
    alone = prepare(job, tmp_path / "one", workers=0, echo=said.append)
    shared = prepare(job, tmp_path / "two", workers=2, echo=said.append)
    meta = json.loads((alone / "dataset.json").read_text())
    assert meta["stats"]["docs"] == 600
    assert len(meta["parts"]) == job.chunks(meta["stats"]["tokens"]) == 32
    for part in meta["parts"]:
        assert pq.read_table(alone / part).equals(pq.read_table(shared / part))
    assert said[0].startswith(f"{job.name}: 600 documents")
    assert prepare(job, tmp_path / "one", echo=said.append) == alone and len(said) == 2


def test_prepared_rows_come_epoch_after_epoch_and_resume(tmp_path, byte_tokenizer):
    t = _train_config(tcfg.TextConfig(path=corpus(tmp_path / "corpus")), byte_tokenizer)
    folder = prepare(jobs(t)[0], tmp_path / "cache", workers=0)
    rows = pa.concat_tables(pq.read_table(p) for p in sorted(folder.glob("*.parquet")))
    stream = batches(t.data, t, 260, cache=tmp_path / "cache")
    epoch = [next(stream) for _ in range(-(-len(rows) // 4) * 2)]
    seen = np.concatenate([b.input_ids for b in epoch])
    first, second = seen[: len(rows)], seen[len(rows) : 2 * len(rows)]
    expected = sorted(map(tuple, np.array(rows["input_ids"].to_pylist())))
    assert sorted(map(tuple, first)) == expected == sorted(map(tuple, second))
    assert not (first == second).all()  # in another order
    again = next(batches(t.data, t, 260, cache=tmp_path / "cache", start=5))
    assert all((a == b).all() for a, b in zip(again[:3], epoch[5][:3], strict=True))
    with pytest.raises(ValueError, match="more than the model's vocab_size of 100"):
        next(batches(t.data, t, 100, cache=tmp_path / "cache"))


def test_a_mixture_has_its_share_of_every_batch(tmp_path):
    assert apportion([3, 1], 4000) == [3000, 1000]
    assert apportion([1, 1, 1], 4) == [2, 1, 1]
    t = _train_config(tcfg.SyntheticConfig(), None, batch=4)
    parts = (tcfg.Weighted(weight=w, source=tcfg.SyntheticConfig()) for w in (1000, 1))
    mix = tcfg.MixtureConfig(sources=tuple(parts))
    with pytest.raises(ValueError, match="no room for a source of weight 0.000999"):
        next(batches(mix, t, 10))


def test_the_same_rows_on_spark(tmp_path, byte_tokenizer):
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    from bux.data import spark

    t = _train_config(tcfg.TextConfig(path=corpus(tmp_path / "corpus")), byte_tokenizer)
    [job] = jobs(t)
    local = prepare(job, tmp_path / "local", workers=0)
    session = (
        SparkSession.builder.master("local[2]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        remote = spark.prepare(job, session, str(tmp_path / "spark"))
        # Prepared, it is returned at once; half written, it is written again.
        assert spark.prepare(job, session, str(tmp_path / "spark")) == remote
        (tmp_path / "spark" / job.name / "dataset.json").unlink()
        assert spark.prepare(job, session, str(tmp_path / "spark")) == remote
    finally:
        session.stop()

    def rows(folder) -> list:
        tables = [pq.read_table(p) for p in sorted(folder.glob("*.parquet"))]
        return sorted(map(json.dumps, pa.concat_tables(tables).to_pylist()))

    assert rows(local) == rows(tmp_path / "spark" / job.name)
    assert remote.endswith(job.name)
