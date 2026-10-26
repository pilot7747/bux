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
smoltalk = chat(path="data/smol-smoltalk/data/train-*.parquet")
```

{bux}`chat` is conversations in the OpenAI format: a list of `{role, content}`
messages per row, `content` a string or a list of `{type: text, text}` parts, whose
texts are joined.

| Field | Default | What it is |
|-------|---------|------------|
| `path` | required | a glob of Parquet files |
| `column` | `messages` | the column of messages |
| `template` | none | the Hub repo or folder whose chat template renders the conversations; none: the tokenizer's own |
| `train_on` | `assistant` | the loss is on every `assistant` reply, the `last` one, or `all` of the conversation |
| `split` | `[0, 1]` | as for `text` |

### The template

A conversation is rendered with a Hugging Face tokenizer's Jinja chat template, as
transformers' `apply_chat_template` renders it: the `chat_template.jinja` of the
tokenizer's repo or folder, or else the `chat_template` of its
`tokenizer_config.json`. bux writes no template of its own, so a conversation reads
as the model's tokenizer has it read. `template` borrows another tokenizer's: a Hub
repo, as `template="HuggingFaceTB/SmolLM-135M-Instruct"` (ChatML), or a folder.

Special tokens written in a message stay text, as in documents
([The tokenizer](#the-tokenizer)): only the template puts them in.

### What the loss is on

With `train_on=assistant`, the targets are every reply, a message whose role is
`assistant`: from where the template's prompt for it ends, what it renders for the
messages before it with `add_generation_prompt`, through the first special token
after that, the one that ends the reply (`<|im_end|>`, `<|eot_id|>`,
`<end_of_turn>`), so that the model learns to stop. The headers, the separators and
every other message are not. `last` takes only the last reply; `all` takes every
token, the template's too. A conversation with no target, as one with no reply, is
dropped.

So the template must render the prompt for each reply as the conversation starts.
ChatML's (Qwen, SmolLM), Llama 3's, Gemma 3's and Phi-4's do; Mistral's moves the
system prompt to the last user message, and DeepSeek-R1's opens a `<think>` in the
prompt that the conversation lacks, so preparing with either fails and says to
borrow another template.

Here is a conversation of a system prompt and two exchanges, in a Parquet file under
`data/chat/` beside a second conversation of a user's message alone. A run of the
smoke model reads it, rendered with Qwen3's template:

```bux
vocab_size = 151936                  # Qwen3's 151669 tokens, padded
tokenizer  = tokenizer(name="Qwen/Qwen3-0.6B")
data       = chat(path="data/chat/*.parquet")
packing    = greedy
seq_len    = 64
```

```console
$ bux prepare chat.bux
chat-eda7df988e: 1 documents, 52 tokens -> 1 rows of 64, 81.2% filled, 15.6% targets, 0 tokens cut (1.2s)
data/packed/chat-eda7df988e
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
positions 44-50   <think>\n\n</think>\n\n6.<|im_end|>                targets
position  51      \n
positions 0-11    12 tokens of padding
```

Ten targets of 64 tokens: the 15.6% the summary reports. Qwen3's template gives the
last reply an empty thought, `<think>\n\n</think>\n\n`, which is learned with it, and
the earlier replies none. `examples/experiments/sft.bux` is a whole fine-tuning run
on SmolTalk.

:::{note}
A borrowed template may write special tokens the tokenizer lacks, which is an error.
GPT-2's tokenizer has no chat template and no `<|im_start|>`: borrow ChatML and add
its tokens after the vocabulary, with
`tokenizer(name="gpt2", added=["<|im_start|>", "<|im_end|>"])`
([The tokenizer](#the-tokenizer)) and
`chat(path=..., template="HuggingFaceTB/SmolLM-135M-Instruct")`.
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

{bux}`tokenizer` is a Hugging Face tokenizer, run by
[gigatoken](https://github.com/marcelroed/gigatoken) at hundreds of megabytes a
second, or by Hugging Face's
[tokenizers](https://github.com/huggingface/tokenizers) where gigatoken cannot run it:
one that is not BPE (WordPiece, Unigram), or one whose byte-level vocabulary lacks
some bytes, as SmolLM2's. The ids are the same; only the speed differs (below).

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

An eval also reads the tokenizer, for the bytes of text each token stands for, and
reports bits per byte beside the loss ([Eval](../training/running.md#eval)).
`tokens` data needs no tokenizer, but a run of shards that sets theirs has them too.
A token's bytes are exact, but for a tokenizer on Hugging Face's tokenizers that is
not byte-level (WordPiece, Unigram): its are what its decoder writes for the token
alone, which leaves out the spaces it writes between words.

Hugging Face's tokenizers is the slower of the two. Encoding 16 MB of text (bux's
docs, in pieces of 4,000 characters) with Qwen3's tokenizer, which both run, took
gigatoken 0.04 s (431 MB/s) and tokenizers 0.56 s (29 MB/s) on an Apple M3 Max with
16 cores, on 2026-10-26.

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

A held-out split of each source of a mixture is a list of evals, which measure each
apart, by loss and bits per byte ([Several held-out sets](../training/running.md#several-held-out-sets)):

```bux
data = 0.7 * web(split=[0, 0.99]) + 0.3 * code(split=[0, 0.99])
eval = [eval(name="web", data=web(split=[0.99, 1])),
        eval(name="code", data=code(split=[0.99, 1]))]
```

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
