# Sources

A run's `data` setting says what it trains on: one source, or a weighted mixture of
sources. Text and chat sources are Parquet files that bux tokenizes and packs before
training. Token shards and random tokens are rows already. This page covers each
kind, the tokenizer, held-out splits, and how mixtures stay deterministic.

```bux
tokenizer = tokenizer(name="Qwen/Qwen3-0.6B")
web       = text(path="data/web/*.parquet")
smoltalk  = chat(path="data/smol-smoltalk/data/train-*.parquet")

data = 0.7 * web(split=[0, 0.99]) + 0.3 * smoltalk
eval = eval(data=web(split=[0.99, 1]))
```

| Source | What it reads | Needs |
|--------|---------------|-------|
| {bux}`text` | plain text, one document per Parquet row | a `tokenizer`; packed before training |
| {bux}`chat` | conversations in the OpenAI format, one per Parquet row | a `tokenizer`; packed before training |
| {bux}`tokens` | flat binary shards of token ids | nothing: already rows |
| {bux}`synthetic` | uniformly random tokens | nothing: already rows |

Paths are globs, relative to the working directory.

## Text

```bux
web  = text(path="data/fineweb-edu/*.parquet", bos=true)
code = text(path="data/stack-edu/*.parquet", column="content", bos=true)
```

{bux}`text` reads plain text, one document per row of the Parquet files:

| Field | Default | What it is |
|-------|---------|------------|
| `path` | required | a glob of Parquet files, relative to the working directory |
| `column` | `text` | the column that holds the text |
| `bos` | `false` | start every document with the tokenizer's `bos` |
| `eos` | `true` | end every document with the tokenizer's `eos`, so that the model learns where text ends |
| `split` | `[0, 1]` | keep the documents whose hash falls in `[start, end)` ([Splits](#splits)) |

Every token of a text document is a target. A document that tokenizes to nothing,
such as an empty or null string, is dropped.

## Chat

```bux
smoltalk = chat(path="data/smol-smoltalk/data/train-*.parquet")
```

{bux}`chat` reads conversations in the OpenAI format. Each row holds a list of
`{role, content}` messages. A message's `content` is a string, or a list of
`{type: text, text}` parts, which are joined.

| Field | Default | What it is |
|-------|---------|------------|
| `path` | required | a glob of Parquet files |
| `column` | `messages` | the column that holds the messages |
| `template` | none | a Hub repo or folder whose chat template renders the conversations; none uses the tokenizer's own |
| `train_on` | `assistant` | where the loss is: every `assistant` reply, the `last` reply, or `all` tokens |
| `split` | `[0, 1]` | as for `text` |

### The template

bux renders a conversation with a Hugging Face tokenizer's Jinja chat template,
exactly as transformers' `apply_chat_template` does. The template is the
`chat_template.jinja` in the tokenizer's repo or folder, or else the `chat_template`
in its `tokenizer_config.json`. bux has no templates of its own, so a conversation is
rendered the way the model's tokenizer expects.

To use another tokenizer's template, set `template` to a Hub repo, such as
`template="HuggingFaceTB/SmolLM-135M-Instruct"` (ChatML), or to a folder.

Special tokens typed inside a message stay plain text, as they do in documents
([The tokenizer](#the-tokenizer)). Only the template inserts special tokens.

### What the loss is on

With `train_on=assistant`, every reply is a target: each message whose role is
`assistant`. A reply's targets start where the template's prompt for it ends, which
is what the template renders for the earlier messages with `add_generation_prompt`.
They run through the first special token after that, the one that ends the reply
(`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`), so that the model learns to stop.
Headers, separators and all other messages are not targets.

- `train_on=last` makes only the last reply a target.
- `train_on=all` makes every token a target, the template's included.
- A conversation with no targets, such as one with no reply, is dropped.

This requires a template that renders the prompt for each reply the same way the
full conversation starts. ChatML's (Qwen, SmolLM), Llama 3's, Gemma 3's and Phi-4's
do. Mistral's does not, because it moves the system prompt to the last user message.
DeepSeek-R1's does not either: it opens a `<think>` in the prompt that the
conversation lacks. Preparing data with either fails, and the error says to borrow
another template.

Here is a run that reads two conversations from a Parquet file under `data/chat/`.
The first has a system prompt and two exchanges; the second is a user message with no
reply. The run renders them with Qwen3's template:

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

The conversation with no reply was dropped, which leaves one document. Its row, by
position:

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

That is 10 targets out of 64 tokens, the 15.6% in the summary line. Qwen3's template
gives the last reply an empty thought, `<think>\n\n</think>\n\n`, which is learned
along with the reply; earlier replies get none. `examples/experiments/sft.bux` is a
complete fine-tuning run on SmolTalk.

:::{note}
A borrowed template may use special tokens the tokenizer lacks, which is an error.
GPT-2's tokenizer, for example, has no chat template and no `<|im_start|>`. To train
it on ChatML, add the template's tokens after the vocabulary
([The tokenizer](#the-tokenizer)) and borrow the template:
`tokenizer(name="gpt2", added=["<|im_start|>", "<|im_end|>"])` and
`chat(path=..., template="HuggingFaceTB/SmolLM-135M-Instruct")`.
:::

## Token shards

```bux
data = tokens(path="data/web/train_*.bin", dtype=uint16, header=1024)
```

{bux}`tokens` reads flat binary shards of token ids through memory maps, with no
preparation step:

| Field | Default | What it is |
|-------|---------|------------|
| `path` | required | a glob of shards, relative to the working directory |
| `dtype` | `uint16` | `uint16` or `uint32` |
| `header` | `0` | bytes before the tokens in every shard (1024 in modded-nanogpt's shards) |

A row is a window of `seq_len` tokens from one shard. Windows overlap by one token,
so every token is a target exactly once, and they come in a new order every epoch.
A row reads as a single text: positions run from 0 to `seq_len - 1`, and every token
is a target, whatever end-of-text tokens the shard contains.

## Random tokens

`data = synthetic` is uniformly random tokens over the model's vocabulary, for smoke
tests and throughput measurements. Each row is drawn from its own seed, so a process
can draw just the rows it holds. Like token shards, each row reads as a single text.

## The tokenizer

```bux
tokenizer = tokenizer(name="unsloth/Meta-Llama-3.1-8B",
                      bos="<|begin_of_text|>", eos="<|end_of_text|>")
```

{bux}`tokenizer` is a Hugging Face tokenizer. bux runs it with
[gigatoken](https://github.com/marcelroed/gigatoken), at hundreds of megabytes per
second. Some tokenizers gigatoken cannot run: those that are not BPE (WordPiece,
Unigram), and byte-level ones whose vocabulary lacks some bytes, such as SmolLM2's.
bux runs those with Hugging Face's
[tokenizers](https://github.com/huggingface/tokenizers) instead. The ids are the
same either way; only the speed differs.

| Field | Default | What it is |
|-------|---------|------------|
| `name` | required | a Hugging Face Hub repo (`gpt2`, `Qwen/Qwen3-0.6B`), or a path to a `tokenizer.json` |
| `bos` | `<\|endoftext\|>` | the special token that `text(bos=true)` starts a document with |
| `eos` | `<\|endoftext\|>` | the special token that `text(eos=true)` ends a document with |
| `added` | `[]` | special tokens appended after the vocabulary, for a template that needs them |

Llama 3's `bos` and `eos` are `<|begin_of_text|>` and `<|end_of_text|>`. GPT-2 has
50257 tokens; padded to a `vocab_size` of 50304, it has room for
`added=["<|im_start|>", "<|im_end|>"]`. Naming a token that is not one of the
tokenizer's special tokens is an error, and the message says how to add it:

```text
the tokenizer has no special token '<|im_start|>'; add it with tokenizer(added=['<|im_start|>']) if the model has room for it
```

**Special tokens written in documents stay text**, as with tiktoken's
`encode_ordinary`: a web page that mentions `<|endoftext|>` does not end there. Only
`bos`, `eos` and chat templates insert special tokens.

A prepared dataset records its tokenizer's vocabulary size. Training fails at once
if that is larger than the model's `vocab_size`.

Evals use the tokenizer too: they count the bytes of text behind each token, and
report bits per byte beside the loss ([Eval](../training/running.md#eval)). `tokens`
data needs no tokenizer, but a run on shards that sets one gets bits per byte as
well. Each token's bytes are exact, with one exception: a tokenizer run by Hugging
Face's tokenizers that is not byte-level (WordPiece, Unigram). For those, a token's
bytes are what the decoder writes for that token alone, which leaves out the spaces
it puts between words.

Hugging Face's tokenizers is the slower of the two. Encoding 16 MB of text (bux's
docs, in pieces of 4,000 characters) with Qwen3's tokenizer, which both can run:

| Tokenizer | Time | Rate |
|-----------|------|------|
| gigatoken | 0.04 s | 431 MB/s |
| Hugging Face's tokenizers | 0.56 s | 29 MB/s |

Measured on an Apple M3 Max with 16 cores, on 2026-10-26.

## Splits

A document's hash is computed from its content: its text, or a conversation's
messages. So `split=[0, 0.99]` and `split=[0.99, 1]` are disjoint and together cover
every document, whatever the tokenizer or the packing. A held-out set needs no
separate file.

```bux
fineweb = text(path="data/fineweb/sample/10BT/*.parquet", bos=true, eos=false)
data    = fineweb(split=[0, 0.99])
eval    = eval(data=fineweb(split=[0.99, 1]), every=125, batches=20)
```

A split is also a deterministic sample: `split=[0, 0.1]` is a tenth of the documents,
the same tenth on any machine.

To evaluate each source of a mixture on its own held-out split, give `eval` a list.
Each eval reports its own loss and bits per byte
([Several held-out sets](../training/running.md#several-held-out-sets)):

```bux
data = 0.7 * web(split=[0, 0.99]) + 0.3 * code(split=[0, 0.99])
eval = [eval(name="web", data=web(split=[0.99, 1])),
        eval(name="code", data=code(split=[0.99, 1]))]
```

## Mixtures

```bux
data = 0.8 * web(split=[0, 0.999]) + 0.2 * code
```

A mixture is a weighted sum of sources: `+` mixes and `*` weights.

- Every batch of `0.7 * web + 0.3 * code` takes 70% of its rows from `web`, rounded.
  Each source gets `round(weight · batch)` rows, by largest remainder, so the counts
  add up to `batch`.
- `web + code` is an even mix.
- Nested mixtures multiply out.
- A source whose share of a batch rounds to zero rows is an error when the run
  starts.

### Determinism

Each source is an endless stream of rows, epoch after epoch, each epoch in a new
order drawn from the run's seed. ([Reading prepared data](pipeline.md#reading-prepared-data)
says how each kind of source orders its rows, and how each process reads only its
own.) Batch `i` holds rows `i·n` to `(i + 1)·n` of each source's stream, where `n` is
that source's share of the batch. So:

- batch `i` of a run depends only on its seed and `i`;
- a run that resumes at step `i` reads exactly what it would have read had it never
  stopped ([Resuming](../training/launching.md#resuming));
- evals draw their batches from a separate stream, independent of training's, and
  read the same batches at every eval.

Under a curriculum or a batch ramp, `n` changes from one stretch of steps to the
next. A source read at the same `seq_len` continues from the row where the previous
stretches left it; at another `seq_len` it is a separate dataset, read from its first
row ([What each phase reads](../training/curriculum.md#what-each-phase-reads)).
