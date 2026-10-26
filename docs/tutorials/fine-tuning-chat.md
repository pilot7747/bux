# Fine-tuning on chat

In this tutorial you train on conversations: a Parquet file of chats in the OpenAI
format, read by `chat(...)`, rendered with a chat template, and learned only where
the assistant speaks. You look at exactly which tokens are targets, change that with
`train_on`, and then read `examples/experiments/sft.bux`, bux's supervised
fine-tuning example.

It builds on [Your own data](your-own-data.md), whose GPT-2 tokenizer it reuses.

:::{note}
bux trains every model from its own initialization: it does not load weights trained
by other frameworks. "Fine-tuning" here is supervised fine-tuning *data*:
conversations, templates and loss masks. `sft.bux` trains a small model from scratch
on SmolTalk's conversations.
:::

## 1. Write some conversations

`chat(...)` reads a column of conversations, each a list of `{role, content}`
messages, as the OpenAI API takes them. `content` is a string, or a list of
`{type: text, text}` parts. Save this as `make_chats.py` and run it:

```python
"""Write conversations in the OpenAI format to data/chats/: a column of messages."""

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def turn(rng: random.Random) -> list[dict]:
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    return [
        {"role": "user", "content": f"What is {a} plus {b}?"},
        {"role": "assistant", "content": f"{a} plus {b} is {a + b}."},
    ]


def conversation(rng: random.Random) -> list[dict]:
    messages = [{"role": "system", "content": "You add numbers."}]
    for _ in range(rng.randint(1, 3)):
        messages += turn(rng)
    return messages


rng = random.Random(0)
chats = [conversation(rng) for _ in range(3000)]
Path("data/chats").mkdir(parents=True, exist_ok=True)
pq.write_table(pa.table({"messages": chats}), "data/chats/sums.parquet")
print(len(chats), "conversations, for example:", chats[0])

# Content may also be a list of parts, as the OpenAI format allows.
parts = [
    [
        {"role": "user", "content": [{"type": "text", "text": "What is 2 plus 2?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "2 plus 2 is 4."}]},
    ]
]
pq.write_table(pa.table({"messages": parts}), "data/chats/parts.parquet")
```

```console
$ python make_chats.py
3000 conversations, for example: [{'role': 'system', 'content': 'You add numbers.'}, {'role': 'user', 'content': 'What is 97 plus 53?'}, {'role': 'assistant', 'content': '97 plus 53 is 150.'}, {'role': 'user', 'content': 'What is 5 plus 33?'}, {'role': 'assistant', 'content': '5 plus 33 is 38.'}]
```

Each conversation has a system prompt and one to three questions with their
answers. The parts go in a file of their own, because Arrow cannot mix strings and
lists in one column; the glob below reads both files.

## 2. The run

Create `mine/chat.bux`:

```bux
# Supervised fine-tuning data: conversations rendered with ChatML, learned where the
# assistant speaks. GPT-2 has no chat template, so this borrows SmolLM's, and none of
# ChatML's tokens, so they are added after its vocabulary.
d_model    = 128
vocab_size = 50304                 # 50257 + 2 added tokens fit under the padding

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn   = gated_mlp(intermediate_size=352, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^4 norm head(tied=true)

tokenizer = tokenizer(name="gpt2", added=["<|im_start|>", "<|im_end|>"])
chats     = chat(path="data/chats/*.parquet", template="HuggingFaceTB/SmolLM-135M-Instruct")
packing   = best_fit(long=drop)    # a conversation cut short may lose its answer

data      = chats(split=[0, 0.95])
eval      = eval(data=chats(split=[0.95, 1]), every=50, batches=4)

optimizer = adamw(lr=3e-3)
schedule  = cosine(warmup=20)
seq_len   = 128
batch     = 8
steps     = 200
log       = log(every=25)
```

What is new:

- `chat(path=..., template=...)` ({bux}`chat`) renders each conversation with a
  Hugging Face tokenizer's chat template, as transformers does: the run's tokenizer's
  own, or another's. GPT-2's has none, so `template` borrows SmolLM's, which is
  ChatML. The column is `messages` unless you say `column="..."`.
- A template writes special tokens, which the tokenizer must have. GPT-2 has none of
  ChatML's, so `added` appends them after its vocabulary, as ids 50257 and 50258;
  `vocab_size = 50304` has room for them. A tokenizer with a chat template of its own,
  as Qwen3's and SmolLM2's, needs neither `template` nor `added`.
- `best_fit(long=drop)` drops a conversation longer than a row rather than split it,
  as a split one may lose its answer. Conversations shorter than a row are packed
  together, each seeing only itself.
- `schedule = cosine(warmup=20)` decays the learning rate to a tenth of its peak at
  the last step.

## 3. Prepare and look at the targets

```console
$ bux prepare mine/chat.bux
chat-afec86b5be: 2,843 documents, 155,873 tokens -> 1,417 rows of 128, 85.9% filled, 21.9% targets, 0 tokens cut (0.6s)
data/packed/chat-afec86b5be
chat-719b1bd2be: 158 documents, 8,851 tokens -> 83 rows of 128, 83.3% filled, 21.3% targets, 0 tokens cut (0.5s)
data/packed/chat-719b1bd2be
```

Only 21.9% of the tokens are targets: the replies. To see which, decode the first
conversation of the first row and bracket the tokens whose `loss_mask` is true. Save
this as `show_chat.py`:

```python
"""Print the first conversation of a packed row, its targets in [brackets]."""

import itertools

import pyarrow.parquet as pq

from bux.data.tokenizer import load
from bux.train.config import TokenizerConfig

tok = load(TokenizerConfig(name="gpt2", added=("<|im_start|>", "<|im_end|>")))
rows = pq.read_table("data/packed/chat-afec86b5be/part-00000.parquet")
row = rows.slice(0, 1).to_pylist()[0]
end = row["position_ids"].index(0, 1)  # where the second conversation starts
pairs = zip(row["input_ids"][:end], row["loss_mask"][:end], strict=True)
for learned, run in itertools.groupby(pairs, key=lambda pair: pair[1]):
    shown = tok.decode([i for i, _ in run])
    print(f"[{shown}]" if learned else shown, end="")
```

```console
$ python show_chat.py
<|im_start|>system
You add numbers.<|im_end|>
<|im_start|>user
What is 10 plus 55?<|im_end|>
<|im_start|>assistant
[10 plus 55 is 65.<|im_end|>]
<|im_start|>user
What is 70 plus 93?<|im_end|>
<|im_start|>assistant
[70 plus 93 is 163.<|im_end|>]
<|im_start|>user
What is 32 plus 95?<|im_end|>
<|im_start|>assistant
[32 plus 95 is 127.<|im_end|>]
```

ChatML writes each message as `<|im_start|>role`, a newline, the content,
`<|im_end|>` and a newline. The loss is on each reply **with the `<|im_end|>` that
ends it**, so that the model learns to stop; the headers, the system prompt and the
user's turns are context only.

## 4. Choose what to learn

`train_on` says which messages are targets:

| `train_on`  | targets                                  | here       |
|-------------|------------------------------------------|------------|
| `assistant` | every assistant reply (the default)      | 21.9%      |
| `last`      | the last assistant reply alone           | 11.0%      |
| `all`       | every token of the conversation          | 84.4%      |

```console
$ bux prepare mine/chat.bux chats.train_on=last
chat-6a5600ec5e: 2,843 documents, 155,873 tokens -> 1,417 rows of 128, 85.9% filled, 11.0% targets, 0 tokens cut (0.6s)
...
$ bux prepare mine/chat.bux chats.train_on=all
chat-e83ed8581f: 2,843 documents, 155,873 tokens -> 1,417 rows of 128, 85.9% filled, 84.4% targets, 0 tokens cut (0.6s)
...
```

`chats.train_on=last` is an amendment of the `chats` definition, from the command
line. A conversation with nothing to learn (no reply, under `assistant` or `last`) is
dropped.

The loss is the mean over every target of the batch, so a row of many short replies
weighs no more than a row of one long one.

## 5. Train

```console
$ bux train mine/chat.bux
remat=auto: none in 1s (none 0.5 GiB in 0s)
chat step 25/200  loss 2.8811  lr 2.99e-03  grad 0.936  6,491 tok/s
chat step 50/200  loss 2.2371  lr 2.82e-03  grad 0.678  7,843 tok/s
chat step 50/200  eval loss 2.2141
chat step 75/200  loss 2.1074  lr 2.42e-03  grad 0.547  8,470 tok/s
chat step 100/200  loss 2.0963  lr 1.88e-03  grad 0.604  8,024 tok/s
chat step 100/200  eval loss 2.1153
chat step 125/200  loss 2.0944  lr 1.30e-03  grad 0.570  8,419 tok/s
chat step 150/200  loss 2.0535  lr 7.82e-04  grad 0.597  8,075 tok/s
chat step 150/200  eval loss 2.0266
chat step 175/200  loss 1.9480  lr 4.26e-04  grad 0.586  8,360 tok/s
chat step 200/200  loss 1.8523  lr 3.00e-04  grad 0.636  8,098 tok/s
chat step 200/200  eval loss 1.8798
```

The loss is over the replies alone. Repeating the question is easy, and the digits of
the sum are the hard part, which 200 steps of a tiny model only begin to learn.

:::{tip}
On the command line, `'tokenizer = tokenizer(name="gpt2")'` keeps the file's
`added`: inside its own definition, a name means the definition it replaces, so this
changes one setting and keeps the others. To drop the added tokens, write `'tokenizer.added = []'`;
then preparing fails at once, as it should:
`the tokenizer has no special token '<|im_start|>'; add it with tokenizer(added=['<|im_start|>']) if the model has room for it`.
:::

## 6. The SFT example

`examples/experiments/sft.bux` is the same recipe at a useful size: a 12-layer,
Qwen3-shaped model trained from scratch on
[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), in Qwen3's
tokens, rendered with Qwen3's own chat template. Without its header comment:

```bux
use llama3_8b

d_model    = 512
vocab_size = 151936                  # Qwen3's 151669 tokens, padded
attn.num_heads    = 8
attn.num_kv_heads = 2
attn.qk_norm      = rmsnorm(eps=1e-6)
ffn.intermediate_size = 1536
model = embed block^12 norm head(tied=true)

tokenizer = tokenizer(name="Qwen/Qwen3-0.6B")
smoltalk  = chat(path="data/smol-smoltalk/data/train-*.parquet")
packing   = best_fit(long=drop)

data      = smoltalk(split=[0, 0.99])
optimizer = adamw(lr=1e-3)
schedule  = cosine(warmup=100)
seq_len   = 2048
batch     = 32
steps     = 2k

eval = eval(data=smoltalk(split=[0.99, 1]), every=250)
```

It borrows Llama 3 8B's block and changes it into Qwen3's shape: 8 query heads and 2
key-value heads, an RMSNorm on queries and keys (`qk_norm`), and a smaller MLP. A
hundredth of the conversations is held out by hash.

```console
$ bux check examples/experiments/sft.bux
d_model=512 vocab_size=151936 layers=51
rmsnorm×25  gqa×12  gated_mlp×12  embed×1  head×1
2,000 steps × 32 × 2048 tokens = 131M tokens
optimizer=adamw(lr=0.001) schedule=cosine(warmup=100)
data=chat(path="data/smol-smoltalk/data/train-*.parquet", split=[0.0, 0.99])
tokenizer=tokenizer(name="Qwen/Qwen3-0.6B") packing=best_fit(long=drop)
$ bux plan examples/experiments/sft.bux
1 run
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
b96ed37410  114M    131M    1.1        759M       1     435MiB      870MiB     0%      sft
```

To train it, download the data with the Hugging Face CLI (from the `huggingface_hub`
package, which bux does not install; `uvx` runs it without installing), then prepare
and train. It is meant for a GPU: 131M tokens of a 114M-parameter model.

```sh
uvx --from huggingface_hub hf download HuggingFaceTB/smol-smoltalk \
    --repo-type dataset --local-dir data/smol-smoltalk
bux prepare examples/experiments/sft.bux
bux launch examples/experiments/sft.bux
```

## What you learned

| you wrote                                  | it does                                       |
|--------------------------------------------|-----------------------------------------------|
| `chat(path=..., template=...)`             | conversations in the OpenAI format, rendered with a tokenizer's chat template |
| `tokenizer(..., added=[...])`              | special tokens after the vocabulary, for a template it lacks |
| `train_on=assistant`, `last`, `all`        | which messages are targets                    |
| `best_fit(long=drop)`                      | drop what does not fit a row, pack the rest   |

[Sources](../data/sources.md) documents `chat` and `text` in full, and
[Packing](../data/packing.md) the packings. Next, [add a layer](add-a-layer.md) of
your own to bux.
