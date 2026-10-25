# Sources

A run's `data` is a source, or a mixture of them. This page covers each kind: raw text
and conversations in Parquet files, which are tokenized and packed before training,
and token shards and random tokens, which are rows already. It also covers the
tokenizer, held-out splits, and how mixtures stay deterministic.

```bux
tokenizer = tokenizer(name="Qwen/Qwen3-0.6B")
web       = text(path="data/web/*.parquet")
smoltalk  = chat(path="data/smol-smoltalk/data/train-*.parquet")

data = 0.7 * web(split=[0, 0.99]) + 0.3 * smoltalk
eval = eval(data=web(split=[0.99, 1]))
```

| Source | What it reads | Needs |
|--------|---------------|-------|
| {bux}`text` | raw text, a document a row of Parquet files | a `tokenizer`; `bux prepare` packs it |
| {bux}`chat` | conversations in the OpenAI format, a conversation a row of Parquet files | a `tokenizer`; `bux prepare` packs it |
| {bux}`tokens` | flat binary shards of token ids | nothing: rows already |
| {bux}`synthetic` | uniformly random tokens | nothing: rows already |

Paths are globs, relative to the working directory.

## Text

```bux
web  = text(path="data/fineweb-edu/*.parquet", bos=true)
code = text(path="data/stack-edu/*.parquet", column="content", bos=true)
```

{bux}`text` is raw text, one document per row of Parquet files:

| Field | Default | What it is |
|-------|---------|------------|
| `path` | required | a glob of Parquet files, relative to the working directory |
| `column` | `text` | the column of text |
| `bos` | `false` | start every document with the tokenizer's `bos` |
| `eos` | `true` | end every document with its `eos`, so that the model learns where text ends |
| `split` | `[0, 1]` | the documents whose hash falls in `[start, end)` ([Splits](#splits)) |

Every token of a text document is a target. A document that tokenizes to nothing, an
empty or a null string, is dropped.

## Chat

```bux
smoltalk = chat(path="data/smol-smoltalk/data/train-*.parquet", template=chatml)
```

{bux}`chat` is conversations in the OpenAI format: a list of `{role, content}`
messages per row, `content` a string or a list of `{type: text, text}` parts, whose
texts are joined.

| Field | Default | What it is |
|-------|---------|------------|
| `path` | required | a glob of Parquet files |
| `column` | `messages` | the column of messages |
| `template` | `chatml` | `chatml` (Qwen, SmolLM) or `llama3` |
| `train_on` | `assistant` | the loss is on every `assistant` reply, the `last` one, or `all` of the conversation |
| `split` | `[0, 1]` | as for `text` |

### What the loss is on

The template renders each message as a header with its role, the message's text, and
the token that ends it:

| Template | Before the first message | Before each message | After each message |
|----------|--------------------------|---------------------|--------------------|
| `chatml` | | `<\|im_start\|>{role}\n` | `<\|im_end\|>`, then `\n` |
| `llama3` | `<\|begin_of_text\|>` | `<\|start_header_id\|>{role}<\|end_header_id\|>\n\n` | `<\|eot_id\|>` |

With `train_on=assistant`, the targets are the text of every message whose role is
`assistant`, with the token that ends it (`<|im_end|>`, `<|eot_id|>`), so that the
model learns to stop. The headers, the separators, and every other message are not.
`last` takes only the last assistant reply; `all` takes every token, the template's
too. A conversation with no target, as one with no assistant reply, is dropped.

Here is a conversation of a system prompt and two exchanges, in a Parquet file under
`data/chat/` beside a second conversation of a user's message alone. A run of the
smoke model reads it:

```bux
vocab_size = 151936                  # Qwen3's 151669 tokens, padded
tokenizer  = tokenizer(name="Qwen/Qwen3-0.6B")
data       = chat(path="data/chat/*.parquet")
packing    = greedy
seq_len    = 64
```

```console
$ bux prepare chat.bux
chat-121c3047c3: 1 documents, 48 tokens -> 1 rows of 64, 75.0% filled, 9.4% targets, 0 tokens cut (0.4s)
data/packed/chat-121c3047c3
```

The conversation with no reply was dropped: one document of the two. The row it made,
by position, with its targets:

```text
positions  0-8    <|im_start|>system\nYou are terse.<|im_end|>\n
positions  9-21   <|im_start|>user\nWhat is 2 + 2?<|im_end|>\n
positions 22-24   <|im_start|>assistant\n
positions 25-27   4.<|im_end|>                                    targets
position  28      \n
positions 29-40   <|im_start|>user\nAnd 3 + 3?<|im_end|>\n
positions 41-43   <|im_start|>assistant\n
positions 44-46   6.<|im_end|>                                    targets
position  47      \n
positions 0-15    16 tokens of padding
```

Six targets of 64 tokens: the 9.4% the summary reports. `examples/experiments/sft.bux`
is a whole fine-tuning run on SmolTalk.

:::{note}
The template's tokens must be special tokens of the tokenizer. GPT-2's tokenizer has
no `<|im_start|>`; add the template's tokens after its vocabulary with
`tokenizer(name="gpt2", added=["<|im_start|>", "<|im_end|>"])`
([The tokenizer](#the-tokenizer)).
:::

## Token shards

```bux
data = tokens(path="data/web/train_*.bin", dtype=uint16, header=1024)
```

{bux}`tokens` reads flat binary shards of token ids through memory maps, with no
preparation:

| Field | Default | What it is |
|-------|---------|------------|
| `path` | required | a glob of shards, relative to the working directory |
| `dtype` | `uint16` | `uint16` or `uint32` |
| `header` | `0` | bytes before the tokens in every shard (1024 in modded-nanogpt's shards) |

A row is a window of `seq_len` tokens of a shard. Windows overlap by a token, so that
every token is a target once, and come in a new order every epoch. A row reads as one
text: positions run from 0 to `seq_len - 1`, and every token is a target, whatever
end-of-text tokens the shard holds.

## Random tokens

`data = synthetic` is uniformly random tokens of the model's vocabulary, for smoke
tests and for measuring throughput. Each row of its stream is drawn from a seed of its
own, so a process draws the rows it holds alone. Its rows read as one text, as token
shards' do.

## The tokenizer

```bux
tokenizer = tokenizer(name="unsloth/Meta-Llama-3.1-8B",
                      bos="<|begin_of_text|>", eos="<|end_of_text|>")
```

{bux}`tokenizer` is a BPE tokenizer run by
[gigatoken](https://github.com/marcelroed/gigatoken), at gigabytes a second:

| Field | Default | What it is |
|-------|---------|------------|
| `name` | required | a Hugging Face Hub repo (`gpt2`, `Qwen/Qwen3-0.6B`), or a path to a `tokenizer.json` |
| `bos` | `<\|endoftext\|>` | the special token `text(bos=true)` starts a document with |
| `eos` | `<\|endoftext\|>` | the special token `text(eos=true)` ends a document with |
| `added` | `[]` | special tokens appended after the vocabulary, for a template it lacks |

Llama 3's `bos` and `eos` are `<|begin_of_text|>` and `<|end_of_text|>`. GPT-2's
50257 tokens, padded to a `vocab_size` of 50304, have room for
`added=["<|im_start|>", "<|im_end|>"]`. A token that is no special token of the
tokenizer is an error that says to add it:

```text
the tokenizer has no special token '<|im_start|>'; add it with tokenizer(added=['<|im_start|>']) if the model has room for it
```

**Special tokens written in documents stay text**, as tiktoken's `encode_ordinary`
keeps them: a web page that mentions `<|endoftext|>` does not end there. Only `bos`,
`eos` and templates put special tokens in.

A prepared dataset records its tokenizer's vocabulary size, and training fails at once
if it is larger than the model's `vocab_size`.

## Splits

A document's hash is of its content: its text, or a conversation's messages. So
`split=[0, 0.99]` and `split=[0.99, 1]` are disjoint and cover everything, whatever
the tokenizer or the packing: a held-out set needs no file of its own.

```bux
fineweb = text(path="data/fineweb/sample/10BT/*.parquet", bos=true, eos=false)
data    = fineweb(split=[0, 0.99])
eval    = eval(data=fineweb(split=[0.99, 1]), every=125, batches=20)
```

A split is also a deterministic sample: `split=[0, 0.1]` is a tenth of the documents,
the same tenth on any machine.

## Mixtures

```bux
data = 0.8 * web(split=[0, 0.999]) + 0.2 * code
```

A mixture is a weighted sum of sources. `+` mixes and `*` weights:

- every batch of `0.7 * web + 0.3 * code` has 70% of its rows from `web`, rounded;
  each source's count is `round(weight · batch)`, by largest remainder, so they add up
  to `batch`;
- `web + code` is an even mix;
- nested mixtures multiply out;
- a source whose share of a batch rounds to no rows is an error when the run starts.

### Determinism

Each source is an endless stream of rows, epoch after epoch, each epoch in a new order
drawn from the run's seed ([Reading prepared data](pipeline.md#reading-prepared-data)
says how each kind orders its rows, and how many processes each read only their own).
Batch `i` holds rows `i·n` to `(i + 1)·n` of each source's stream, `n` its share of
the batch. So:

- batch `i` of a run depends only on its seed and `i`;
- a run that resumes at step `i` reads exactly what it would have read had it never
  stopped ([Resuming](../training/launching.md#resuming));
- an eval's batches are drawn from a stream of their own, independently of
  training's, and are the same at every eval.
