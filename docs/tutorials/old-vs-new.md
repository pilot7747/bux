(tutorial-old-vs-new)=
# Old against new at the same serving cost

Is a 2026 architecture better than a 2022 one? At the same number of parameters the
question is unfair to a mixture of experts, which holds many parameters and uses few
of them per token. At the same training FLOPs it ignores what a model costs once it
is trained. This tutorial compares the two at the **same serving cost**, under the load
they are built for, agentic coding: long prompts, a few turns each, many sessions at
once. Then it asks which one predicts code better, on a scaling ladder of small models
extrapolated to the two at full size.

It is the advanced tutorial: it runs on a node of 8 GPUs, not on a laptop, and puts
together pieces the others introduce one by one (sweeps and labels, searches,
curricula, evals, exports, vLLM). The steps:

1. Write the two families, each sized by one number.
2. Price them: export a family at several sizes without weights, time each in vLLM
   under an agentic load, and find the 2026 model that serves as fast as a 2022 7B.
3. Get the data: code, whole repositories for long context, and web text.
4. Tune the learning rate of each family on its small rungs, by search.
5. Train both ladders.
6. Fit a scaling law to each and extrapolate to the two targets.
7. Read the result, and what it does not say.

Every number on this page was measured as written, on nodes of 8 NVIDIA B300s, on
2026-10-26 and 27; [What it takes](#what-it-takes) says how long each step takes on one.

## 1. Two families

A family is a `.bux` file whose model a single number sizes: `d_model`. Everything
else follows from it, so that a ladder of small models and its full-size target are
the same design at different widths.

The **2022 family** is the decoder of OPT, GPT-NeoX and Pythia, in
`examples/experiments/old_vs_new/old_2022.bux`:

```bux
use agentic_code

d_model    = 4096
vocab_size = 50304                       # GPT-NeoX's 50277 tokens, padded

norm  = layernorm(eps=1e-5, bias=true)
attn  = gqa(num_heads=d_model // 128, num_kv_heads=d_model // 128, bias=true,
            rope=rope(base=10000))
ffn   = mlp(intermediate_size=4 * d_model, activation=gelu, bias=true)
block = [norm attn] [norm ffn]

layers = d_model // 128
model  = embed block^layers norm head

budget       = 129B
batch        = 1792
optimizer.lr = 1.5e-3
mesh         = mesh(fsdp=8)
```

Multi-head attention (as many key and value heads as query heads), RoPE at base 10k on
every channel, LayerNorms with biases, a GELU MLP four times as wide as the stream. A
layer for every 128 of the width, so `d_model = 4096` is OPT-6.7B's shape: 32 layers of
32 heads, 6.44B parameters besides the embeddings.

The **2026 family** is laid out as Qwen3-Next is, in
`examples/experiments/old_vs_new/new_2026.bux`:

```bux
use agentic_code

d_model    = 3072
vocab_size = 50304

norm    = rmsnorm(zero_centered=true)
delta   = gdn(num_heads=d_model // 128, num_v_heads=d_model // 64, head_dim=128)
attn    = gqa(num_heads=d_model // 128, num_kv_heads=2, head_dim=256, gated=true,
              qk_norm=norm, rope=rope(base=10000000, fraction=0.25))
experts = moe(num_experts=256, top_k=8, num_shared_experts=1, shared_expert_gate=true,
              aux_loss_coef=1e-3,
              expert=gated_mlp(intermediate_size=d_model // 4, activation=swish))

layer(mixer) = [norm mixer] [norm experts]
D = layer(delta)
A = layer(attn)

groups = d_model // 256
model  = embed "DDDA"^groups norm head

budget       = 131B
batch        = 1792
optimizer.lr = 2.6e-3
mesh         = mesh(fsdp=64)
```

Three Gated DeltaNet layers to one attention layer; the attention has 2 key and value
heads of 256 dims, QK-norm, a gate on its output and RoPE on a quarter of its channels;
after each mixer a mixture of 256 experts a quarter of the width wide, 8 of them and a
shared one for every token, as DeepSeek-V3 splits its experts. A layer for every 64 of
the width.

Each family file is also its target's run: the recipe of step 3 (`use agentic_code`),
20 tokens for every parameter a token goes through, a batch by the law of step 5, and
the learning rate step 4 finds. The ladders of step 5 use the files and set each of
these for every rung. Both read the same tokenizer, GPT-NeoX's, so that only the
architecture differs (bits per byte would compare across tokenizers anyway), and both
start from bux's default init, every weight matrix from a normal of 0.02, as OPT and
Qwen do ([Initialization](../architectures/initialization.md)). `bux check` reads
either:

```console
$ bux check examples/experiments/old_vs_new/new_2026.bux
d_model=3072 vocab_size=50304 layers=195
rmsnorm×97  moe×48  gdn×36  gqa×12  embed×1  head×1
35,694 steps = 131B tokens
from step 0: 1792 × 2048 tokens
phase 1, from step 32,125: 224 × 16384 tokens, data=0.5 * text(path="data/stack-edu/repos/train/*.parquet") + 0.3 * text(path="data/stack-edu/files/train/*.parquet") + 0.2 * text(path="data/fineweb-edu/sample/10BT/*.parquet", split=[0.0, 0.999])
optimizer=adamw(lr=0.0026) schedule=wsd(warmup=0.05, shape=sqrt)
data=0.6 * text(path="data/stack-edu/files/train/*.parquet") + 0.4 * text(path="data/fineweb-edu/sample/10BT/*.parquet", split=[0.0, 0.999])
tokenizer=tokenizer(name="EleutherAI/gpt-neox-20b") packing=best_fit
```

`bux plan` prices the two targets' training:

```console
$ bux plan examples/experiments/old_vs_new/old_2022.bux | tail -1
056601a566  6.86B   129B    18.8       43.5B      8     3.19GiB     6.39GiB    0%      old_2022
$ bux plan examples/experiments/old_vs_new/new_2026.bux | tail -1
3354471942  91.1B   131B    1.4        43.7B      64    5.3GiB      10.6GiB    0%      new_2026
```

43.5B and 43.7B FLOPs a token: the two cost the same to train a token, as they pass it
through as many parameters. Neither trains here; the ladders stand in for them.

What a token costs is not what a model holds. The script `fit.py` of step 6 counts
both, the parameters a token goes through (an MoE's routed experts by 8 of 256)
besides the embeddings and the head:

| `d_model` | 2022: all | 2022: besides embeddings | 2026: all | 2026: active, besides embeddings |
|-----------|-----------|--------------------------|-----------|----------------------------------|
| 512       | 64M       | 12.6M                    | 474M      | 32.1M                            |
| 1024      | 204M      | 101M                     | 3.47B     | 248M                             |
| 2048      |           |                          | 27.1B     | 1.95B                            |
| 3072      |           |                          | 91.1B     | 6.54B                            |
| 4096      | 6.86B     | 6.44B                    |           |                                  |

At `d_model = 3072` the 2026 model holds 13 times the 2022 7B's parameters, and a
token goes through as many of them as through the 7B's.

## 2. What a model costs to serve

A model's serving cost is the GPU time a session takes, at a load that keeps the GPU
busy. The load here is an agent at work on a repository, a session of 4 turns:

- a system prompt of 2k tokens that every session shares, then 14k tokens of the
  session's own context, the repository's files: 16k tokens before the first turn;
- each turn the model writes 512 tokens, and the next turn reads them back with a
  tool's result of 2k tokens appended, so the last prompt is 24k tokens;
- every prompt starts with the one before, so vLLM's prefix cache serves all of it but
  the new tokens, as long as the session's cache was not evicted;
- many sessions at once: a closed loop of C sessions, a new one starting as one ends.

`scripts/old_vs_new/agentic.py` runs that load against vLLM's engine in-process, and
prints a JSON line for each concurrency. It needs no checkpoint: the weights are
random (vLLM's `load_format="dummy"`), and only the time counts. `bux export` of a
`.bux` file writes what vLLM needs, the model's code and config without weights
([Exporting an architecture](../export/index.md#exporting-an-architecture)), so a
family exports at any width in seconds, a 116B model too:

```console
$ bux export examples/experiments/old_vs_new/old_2022.bux exported/old-4096
examples/experiments/old_vs_new/old_2022.bux: 6.86B parameters exported to exported/old-4096, without weights
$ for d in 1536 2048 2560 3072 3328; do
>   bux export examples/experiments/old_vs_new/new_2026.bux exported/new-$d "d_model = $d"
> done
examples/experiments/old_vs_new/new_2026.bux: 11.5B parameters exported to exported/new-1536, without weights
examples/experiments/old_vs_new/new_2026.bux: 27.1B parameters exported to exported/new-2048, without weights
examples/experiments/old_vs_new/new_2026.bux: 52.8B parameters exported to exported/new-2560, without weights
examples/experiments/old_vs_new/new_2026.bux: 91.1B parameters exported to exported/new-3072, without weights
examples/experiments/old_vs_new/new_2026.bux: 116B parameters exported to exported/new-3328, without weights
```

vLLM and bux pin their own PyTorch, so vLLM gets an environment of its own, with bux's
plugin ([Serving in vLLM](../export/vllm.md)); install it from outside the checkout,
where `uv pip` would take bux's own settings:

```sh
cd /tmp && uv venv --python 3.12 vllm-env
uv pip install --python vllm-env/bin/python vllm==0.32.0 ~/bux/src/bux_vllm
cd ~/bux
```

Then each model on a GPU of its own, at 8, 16, 32 and 64 sessions at once, twice as
many sessions in all as at once:

```sh
export PATH=/tmp/vllm-env/bin:$PATH              # its ninja, for FlashInfer's kernels
moe='--engine moe_backend="triton"'
gpu=0
for m in old-4096 new-1536 new-2048 new-2560 new-3072 new-3328; do
  CUDA_VISIBLE_DEVICES=$gpu python scripts/old_vs_new/agentic.py exported/$m \
    --concurrency 8,16,32,64 --sessions 2 $moe &
  gpu=$((gpu + 1))
done; wait
```

`moe_backend="triton"` is for the B300: vLLM's default for an MoE there, FlashInfer's
CUTLASS kernels, fails to build ([Serving in vLLM](../export/vllm.md#installing)).
A line of the output, the 2026 model at `d_model = 3072`, 16 sessions at once:

```json
{"tag": "new-3072", "concurrency": 16, "sessions": 32, "seconds": 86.45, "sessions_per_hour": 1332.5, "output_tok_s": 758.0, "prompt_tok_s": 29942.6, "prefilled_tok_s": 8536.1, "cache_hit": 0.7149, "ttft_p50": 0.387, "ttft_p90": 2.252, "tpot_p50_ms": 18.45, "kv_blocks": 3589, "block_size": 816}
```

Sessions an hour, one GPU each (the 2022 7B also at 12, 20 and 24 sessions at once,
to find its best):

| Model | parameters (active) | 8 at once | 16 | 32 | 64 | time a token, 8 at once |
|-------|---------------------|-----------|----|----|----|-------------------------|
| 2022, `d_model = 4096` | 6.86B | 892 | 992 | 873 | 881 | 15.2 ms |
| 2026, 1536 | 11.5B (0.83B) | 2,672 | 4,338 | 5,736 | 7,433 | 4.0 ms |
| 2026, 2048 | 27.1B (1.95B) | 1,817 | 2,709 | 3,647 | 4,808 | 6.7 ms |
| 2026, 2560 | 52.8B (3.80B) | 1,256 | 1,808 | 2,425 | 3,218 | 9.9 ms |
| **2026, 3072** | **91.1B (6.54B)** | **894** | 1,332 | 1,776 | 2,335 | 14.0 ms |
| 2026, 3328 | 116B (8.31B) | 748 | 1,140 | 1,496 | 1,052 | 16.6 ms |

What sets each model's cost is what it keeps for a session:

- **The 2022 7B is bound by its KV cache.** Multi-head attention keeps 512 KB a token
  (32 layers, keys and values of 4096 channels, in bfloat16), so the 234 GB vLLM
  leaves it beside the weights hold 463k tokens: 19 sessions of 24k. It serves most,
  1,012 sessions an hour, at 20 at once (956 at 12, 948 at 24). Past that the sessions no longer fit
  between their turns: vLLM evicts the prefix of one to serve another, the cache hit
  rate falls from 75% to 10%, and every turn prefills its whole prompt again; at 64 at
  once the first token waits 45 s.
- **The 2026 model is bound by its weights.** It keeps 2 key and value heads of 256
  in a quarter of its layers and a fixed state in the rest: 24 KB a token, a
  twenty-first of the 7B's, and its cache holds 2.9M tokens at `d_model = 3072`. Its
  cost is reading its weights each step, shared by every session in the batch, so
  the more sessions at once the cheaper each: it serves 2.6 times as many at 64 at once as at 8. At 3328 the weights leave
  room for 1.3M tokens of state, and at 64 at once it thrashes as the 7B does.

At 8 sessions at once, the 2026 model at `d_model = 3072` serves as many sessions an
hour as the 2022 7B (894 against 892), each token as fast (14.0 ms against 15.2); at
every load above, more. So it is the target: **as fast as the 7B at the lightest load,
and faster at every other**, which favors the 7B, since a model as cheap as it under
a heavy load would be larger still. It also passes as many parameters per token as
the 7B (6.54B against 6.44B, besides the embeddings), so the two cost about the same
to train a token, too. Its first token comes later at light load (0.39 s against
0.14 s), and a little less of each prompt comes from the cache (71% against 75%): vLLM caches a hybrid's
prefixes in blocks of 816 tokens, where its recurrent state is saved, against the
7B's 16.

## 3. The data

The models are for code, so the data is mostly code, and the long context they serve
is repositories. `scripts/old_vs_new/stack_edu.py` downloads
[Common Pile's Stack v2 Edu](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered),
source files from openly licensed repositories, filtered for educational value
(Kandpal et al., 2025), and writes Parquet files for bux:

```console
$ python scripts/old_vs_new/stack_edu.py data/stack-edu
shard 0 (markdown): 345963 files (29s)
shard 18 (c): 600090 files (55s)
shard 20 (cpp): 642038 files (57s)
...
shard 84 (python repos only): 928213 files (87s)
shard 85 (ruby): 888182 files (87s)
...
bucket 0: 88397 repos, 6828 + 77 long (1082M chars) (101s)
bucket 1: 88438 repos, 6759 + 44 long (1078M chars) (101s)
...
```

On a node of 160 cores and a fast link, the 26 shards (24 GB) download and convert
in under two minutes.

- `files/train/` and `files/eval/`: a source file a row, in 14 languages, Python
  most: 13.7M files to train on, 137k held out.
- `repos/train/` and `repos/eval/`: a Python repository a row, its files in path
  order, each after a line `--- path ---`, for the repositories of at least 64k
  characters (about 16k tokens): 108,935 to train on, 1,051 held out. The shards hold
  files in no order, so the script puts each repository together from all twelve
  Python shards.

A repository is held out whole, its files and its document, when the hash of its name
falls in the last 1%: no file the evals read is trained on, even as part of a
repository. Web text is FineWeb-Edu's 10BT sample:

```sh
hf download HuggingFaceFW/fineweb-edu --repo-type dataset --include "sample/10BT/*" --local-dir data/fineweb-edu
```

`examples/experiments/recipes/agentic_code.bux` is the training recipe both ladders
use, a module with no model:

```bux
tokenizer  = tokenizer(name="EleutherAI/gpt-neox-20b")
web        = text(path="data/fineweb-edu/sample/10BT/*.parquet")
code       = text(path="data/stack-edu/files/train/*.parquet")
code_eval  = text(path="data/stack-edu/files/eval/*.parquet")
repos      = text(path="data/stack-edu/repos/train/*.parquet")
repos_eval = text(path="data/stack-edu/repos/eval/*.parquet")
packing    = best_fit

data    = 0.6 * code + 0.4 * web(split=[0, 0.999])
seq_len = 2048
batch   = 256
budget  = 10B                            # tokens
steps   = budget // (batch * seq_len)

long       = 0.5 * repos + 0.3 * code + 0.2 * web(split=[0, 0.999])
curriculum = phase(at=0.9, seq_len=16384, batch=batch // 8, data=long)

optimizer = adamw(lr=3e-3, weight_decay=0.1)
schedule  = wsd(warmup=0.05, decay=0.2, shape=sqrt)
loss      = loss(z=1e-4)

eval = [eval(name="code", data=code_eval, every=2k, batch=64, batches=40),
        eval(name="web", data=web(split=[0.999, 1]), every=2k, batch=64, batches=40),
        eval(name="repo", data=repos_eval, every=2k, seq_len=16384, batch=8, batches=16)]
objective = 0.5 * code_bpb + 0.3 * repo_bpb + 0.2 * web_bpb
```

- **A long-context phase.** Every run trains its last tenth of tokens on rows of 16k
  tokens, half of them whole repositories, as many tokens a step as before
  ([Curriculum](../training/curriculum.md)). It falls inside the cooldown of the
  warmup-stable-decay schedule, as long-context extensions usually do.
- **Three held-out sets.** Code files and web text, 5.2M tokens each in rows of 2k,
  and whole repositories, 2.1M tokens in rows of 16k: an eval with a `seq_len` of its
  own ([Long rows](../training/running.md#long-rows)) holds the model to the length it
  serves. Each eval has a `batch` of its own too, so that every rung, whatever its
  batch, is measured on the same rows.
- **One objective.** What the searches minimize and the ladders fit: bits per byte,
  half of them on code files, 30% on repositories, 20% on web text.

`bux prepare` tokenizes and packs every dataset the runs read, at each `seq_len`
they read it, once:

```console
$ bux prepare examples/experiments/old_vs_new/old_ladder.bux
...
text-33e9de98ad: 108,935 documents, 5,359,706,785 tokens -> 327,911 rows of 16384, 99.8% filled, 99.8% targets, 0 tokens cut (41.0s)
text-93d98c2ecb: 13,663,938 documents, 16,359,620,446 tokens -> 999,087 rows of 16384, 99.9% filled, 99.9% targets, 0 tokens cut (122.2s)
text-bb80de4655: 9,662,259 documents, 9,909,527,499 tokens -> 605,207 rows of 16384, 99.9% filled, 99.8% targets, 0 tokens cut (75.5s)
text-87a6560c94: 136,882 documents, 168,692,022 tokens -> 82,400 rows of 2048, 100.0% filled, 99.9% targets, 0 tokens cut (6.2s)
text-000e665c59: 9,842 documents, 9,928,901 tokens -> 4,853 rows of 2048, 99.9% filled, 99.8% targets, 0 tokens cut (25.2s)
text-bc3bcef6e3: 1,051 documents, 50,093,111 tokens -> 3,070 rows of 16384, 99.6% filled, 99.6% targets, 0 tokens cut (4.4s)
```

16.4B tokens of code files and 9.9B of web text to train on, packed at 2k and, for
the long phase, at 16k, and 5.4B of whole repositories at 16k; held out, 169M tokens
of code, 10M of web text and 50M of repositories. Eight datasets in 8 minutes on 150 workers, and
`new_ladder.bux` reads the same ones. The largest rung trains on 9.6B tokens, so no
row is read twice.

## 4. A learning rate for each family

A ladder is fair only if each rung trains at its own best learning rate: one rate for
both families would favor whichever it suits, and one rate for every width would favor
the widths it suits. bux's searches find it ([Searching
hyperparameters](../training/searching.md)). A setting written as a distribution makes
a run a search, and the command line can write it, so the ladder files need no edit:
a rung of a ladder with its `optimizer.lr` a dimension is a search of that rung.

```sh
bux launch examples/experiments/old_vs_new/old_ladder.bux --only "[d512]" \
  'optimizer.lr = log_uniform(1e-3, 3e-2)' 'search = search(trials=12)'
```

Twelve trials by the default method, a Gaussian process: quasi-random points first,
then the points of most expected improvement, each a whole run of the rung, a GPU
each. It ends with its table, which `bux results` prints again with the same
arguments:

```console
search old_ladder[d512]: 12 of 12 trials; objective = 0.5 * code_bpb + 0.3 * repo_bpb + 0.2 * web_bpb; best 0.8948, trial 10
trial  rung  objective  run
10     0/0   0.8948     old_ladder[d512, optimizer.lr=0.00104]
1      0/0   0.8952     old_ladder[d512, optimizer.lr=0.00202]
7      0/0   0.9032     old_ladder[d512, optimizer.lr=0.00304]
8      0/0   0.9097     old_ladder[d512, optimizer.lr=0.00313]
6      0/0   0.9101     old_ladder[d512, optimizer.lr=0.00388]
3      0/0   0.9184     old_ladder[d512, optimizer.lr=0.00474]
11     0/0   0.9369     old_ladder[d512, optimizer.lr=0.00589]
9      0/0   0.9385     old_ladder[d512, optimizer.lr=0.00591]
4      0/0   0.9932     old_ladder[d512, optimizer.lr=0.00724]
0      0/0   2.5959     old_ladder[d512, optimizer.lr=0.0111]
5      0/0   2.8887     old_ladder[d512, optimizer.lr=0.013]
2      0/0   3.0809     old_ladder[d512, optimizer.lr=0.0259]
```

**The best trial is at the edge of the range**, 1.04e-3 against a floor of 1e-3, so
the range is too small. Another range is a search of its own, which keeps the first
one's trials where they are:

```console
$ bux launch examples/experiments/old_vs_new/old_ladder.bux --only "[d512]" 'optimizer.lr = log_uniform(3e-4, 3e-3)' 'search = search(trials=8)'
...
search old_ladder[d512]: 8 of 8 trials; objective = 0.5 * code_bpb + 0.3 * repo_bpb + 0.2 * web_bpb; best 0.8899, trial 0
trial  rung  objective  run
0      0/0   0.8899     old_ladder[d512, optimizer.lr=0.00153]
6      0/0   0.8942     old_ladder[d512, optimizer.lr=0.00204]
4      0/0   0.8951     old_ladder[d512, optimizer.lr=0.00115]
2      0/0   0.8971     old_ladder[d512, optimizer.lr=0.00272]
3      0/0   0.8977     old_ladder[d512, optimizer.lr=8.6e-4]
7      0/0   0.9123     old_ladder[d512, optimizer.lr=6.45e-4]
1      0/0   0.9295     old_ladder[d512, optimizer.lr=4.83e-4]
5      0/0   0.9521     old_ladder[d512, optimizer.lr=3.63e-4]
```

The bottom is at 1.5e-3, and it is flat: 1e-3 to 2.7e-3 is within 1% of the best.
Two trials 1% apart in learning rate, 2.02e-3 and 2.04e-3, came 0.001 apart: about
what two runs at the same point differ by, and the resolution of every comparison on
this page.

**The 2022 family diverges.** From 7e-3 up its runs blow up mid-way, and end near 3
bits per byte, where the bottom is under 0.9. A diverged trial is information, but a
costly kind for the Gaussian process, which models the objective as smooth: at d768,
8 of its 12 trials landed above 6.5e-3, five of them its own picks after the first had
diverged, around the cliff's edge. Once a search has found a cliff, cap the next
range below it, as the second d768 search here does (`log_uniform(3e-4, 1.5e-3)`).

The other searches, the same way:

| Rung | active parameters | trials | best trial | its objective | vertex |
|------|-------------------|--------|------------|---------------|--------|
| 2022, d512 | 12.6M | 12 + 8 | 1.53e-3 | 0.8899 | 1.54e-3 |
| 2022, d768 | 42.5M | 12 + 6 | 1.32e-3 | 0.7127 | 1.58e-3 |
| 2022, d1024 | 101M | 8 + 1 | 2.26e-3 | 0.6268 | 1.79e-3 |
| 2026, d512 | 32.1M | 12 | 2.31e-3 | 0.6715 | 2.20e-3 |
| 2026, d768 | 106M | 9, halving | 1.24e-3 | 0.5594 | flat: 1.2e-3 to 3e-3 within 0.07% |

A rung's best learning rate is the vertex of a parabola in log learning rate through
its trials within 3% of its best: the bottom is flat to within the runs' noise, so the
best trial alone could sit anywhere on it.

- **For the 2022 family the best learning rate does not move with width**: 1.54e-3 at
  d512, 1.58e-3 at d768, 1.79e-3 at d1024, a power law of exponent 0.07, which is
  none within the noise. Under the standard parameterization it falls with
  width when the init does (LeCun's, $1/\sqrt{d}$), but bux's default init is a normal
  of 0.02 at every width ([Initialization](../architectures/initialization.md)),
  which for these widths is the smaller one, and the optimum stays put. The wider
  rungs, d1280 and d1536, trained at 1.57e-3 and 1.43e-3, within 5% of it: on a bottom
  this flat, far less than the noise between two runs.
- **The 2026 family does not diverge.** Its bottom spans 1e-3 to 7e-3 within 0.5%,
  and at 2.6e-2, eleven times its best, it loses 7.5%. It has QK-norm and a gate on
  attention's output, which were introduced for stable training at high learning
  rates (Dehghani et al., 2023; Qiu et al., 2025), but a ladder of two whole families
  cannot say which of its differences does it.

The 2026 d768 search halves ([Halving](../training/searching.md#halving)): nine
trials of a third of the run, then the best three in full, so that a search of a rung
of 2 GPUs and 79 minutes a run costs six runs, not nine:

```console
search new_ladder[d768]: 9 of 9 trials; objective = 0.5 * code_bpb + 0.3 * repo_bpb + 0.2 * web_bpb; best 0.5594, trial 1
trial  rung  objective  run
1      1/1   0.5594     new_ladder[d768, optimizer.lr=0.00124]
8      1/1   0.5598     new_ladder[d768, optimizer.lr=0.00265]
3      1/1   0.5601     new_ladder[d768, optimizer.lr=0.00298]
6      0/1   0.6353     new_ladder[d768, optimizer.lr=0.00195, steps=2696]
4      0/1   0.6380     new_ladder[d768, optimizer.lr=0.00462, steps=2696]
0      0/1   0.6423     new_ladder[d768, optimizer.lr=0.00716, steps=2696]
5      0/1   0.6438     new_ladder[d768, optimizer.lr=0.00844, steps=2696]
7      0/1   0.6481     new_ladder[d768, optimizer.lr=6e-4, steps=2696]
2      0/1   0.6568     new_ladder[d768, optimizer.lr=0.0172, steps=2696]
```

The three that trained in full, from 1.2e-3 to 3e-3, are 0.0007 apart: within the
noise, so this rung has no vertex to speak of, only a bottom that holds both searched
widths' optima. The 2026 ladder trains its wider rungs at 2.6e-3, inside it at both.

The ladder files write each searched rung's learning rate as its best trial's, and the
wider rungs' as above, so that the ladders train what the searches found even where
no search was run.

## 5. The ladders

A ladder is a family at a few widths, each trained as its target would be: on 20
tokens for every active parameter besides the embeddings. The targets are 6.44B and
6.54B of them, so both would train on about 130B tokens, which keeps the comparison at
the same data too. `examples/experiments/old_vs_new/old_ladder.bux`:

```bux
use old_2022
use agentic_code

d_model      = d512: 512  | d768: 768  | d1024: 1024 | d1280: 1280 | d1536: 1536
budget       = d512: 252M | d768: 850M | d1024: 2.02B | d1280: 3.94B | d1536: 6.8B
batch        = d512: 32   | d768: 64   | d1024: 128  | d1280: 192  | d1536: 256
optimizer.lr = d512: 1.53e-3 | d768: 1.32e-3 | d1024: 2.26e-3 | d1280: 1.57e-3 | d1536: 1.43e-3
mesh         = d512: mesh(data=1) | d768: mesh(data=1) | d1024: mesh(data=2) | d1280: mesh(data=4) | d1536: mesh(data=8)
```

The labels zip the five settings into one axis ([Labels zip
axes](../language/sweeps.md#labels-zip-axes)): five runs, not 5^5. The batch follows the
batch-size law of DeepSeek LLM (DeepSeek-AI, 2024), `B = 0.292 C^0.327` tokens for C
FLOPs, rounded to 32 rows, and the mesh gives the wider rungs more GPUs. `new_ladder.bux` is the same for the 2026 family:

```bux
use new_2026
use agentic_code

d_model      = d512: 512  | d768: 768   | d1024: 1024 | d1280: 1280
budget       = d512: 642M | d768: 2.12B | d1024: 4.96B | d1280: 9.62B
batch        = d512: 64   | d768: 128   | d1024: 192   | d1280: 320
optimizer.lr = d512: 2.31e-3 | d768: 1.24e-3 | d1024: 2.6e-3 | d1280: 2.6e-3
mesh         = d512: mesh(data=1) | d768: mesh(data=2) | d1024: mesh(fsdp=8) | d1280: mesh(fsdp=8)
```

The 2026 model's MoE holds 3.5B parameters at `d_model = 1024` and 6.7B at 1280: a
copy a GPU of them and of AdamW's state, 16 bytes a parameter, does not leave room for
the 16k phase's activations (`d1024` ran out of memory on 8 copies as the phase began),
so those rungs split them over the 8 GPUs by FSDP ([Scale out](scale-out.md)). A mesh
is not in a run's id, so it changes nothing the run computes. `bux plan` shows the
four:

```console
$ bux plan examples/experiments/old_vs_new/new_ladder.bux
4 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
d805b0b18f  474M    642M    1.4        531M       1     1.76GiB     3.53GiB    0%      new_ladder[d512]
a2652a6bf8  1.5B    2.12B   1.4        1.2B       2     5.59GiB     11.2GiB    0%      new_ladder[d768]
14fc2c245e  3.47B   4.96B   1.4        2.33B      8     1.62GiB     3.23GiB    0%      new_ladder[d1024]
c727d3ecc8  6.7B    9.62B   1.4        4.04B      8     3.12GiB     6.24GiB    0%      new_ladder[d1280]
```

`tok/param` counts all the parameters, the idle experts too; 20 a token's are the
budget. Without the searches, launch both ladders whole, each rung at the learning rate
its file gives:

```sh
bux launch examples/experiments/old_vs_new/old_ladder.bux
bux launch examples/experiments/old_vs_new/new_ladder.bux
```

After the searches, a searched rung's best trial is already its point: it has the
rung's id, though not its name, so `bux launch` would train it again in a directory of
the ladder's name ([Run ids](../training/running.md#run-ids)). The fit of step 6 reads
every run of a ladder's name, the trials too, and takes each rung's best; launch only
the rungs no search trained:

```sh
bux launch examples/experiments/old_vs_new/old_ladder.bux --only "[d1280]"
bux launch examples/experiments/old_vs_new/old_ladder.bux --only "[d1536]"
bux launch examples/experiments/old_vs_new/new_ladder.bux --only "[d1024]"
bux launch examples/experiments/old_vs_new/new_ladder.bux --only "[d1280]"
```

Each takes the GPUs of its mesh, and the next waits for room. On one node of 8 B300s:

| Ladder | rungs at once | wall-clock |
|--------|---------------|------------|
| 2022 | d512, d768, d1024 and d1280 side by side (8 GPUs), then d1536 (8) | 1 h 53 min |
| 2026 | d512 and d768 (3 GPUs), then d1024 (8), then d1280 (8) | 7 h 02 min |

Each rung's run, and its held-out bits per byte after its last step:

| Rung | GPUs | time | active parameters | objective | code | repositories, 16k | web |
|------|------|------|-------------------|-----------|------|-------------------|-----|
| 2022, d512 | 1 | 7 min | 12.6M | 0.8899 | 0.8051 | 0.7599 | 1.2971 |
| 2022, d768 | 1 | 24 min | 42.5M | 0.7127 | 0.6253 | 0.5815 | 1.1281 |
| 2022, d1024 | 2 | 37 min | 101M | 0.6268 | 0.5406 | 0.4976 | 1.0363 |
| 2022, d1280 | 4 | 52 min | 197M | 0.5722 | 0.4891 | 0.4482 | 0.9659 |
| 2022, d1536 | 8 | 61 min | 340M | 0.5345 | 0.4539 | 0.4138 | 0.9173 |
| 2026, d512 | 1 | 27 min | 32.1M | 0.6715 | 0.5908 | 0.5253 | 1.0926 |
| 2026, d768 | 2 | 79 min | 106M | 0.5594 | 0.4804 | 0.4249 | 0.9588 |
| 2026, d1024 | 8 | 98 min | 248M | 0.5009 | 0.4256 | 0.3727 | 0.8814 |
| 2026, d1280 | 8 | 245 min | 481M | 0.4597 | 0.3874 | 0.3370 | 0.8245 |

The ladders alone already compare the families: at about 100M active parameters, the
2026 rung's objective is 11% under the 2022 one's (0.5594 at 106M against 0.6268 at
101M), and the 2026 d768 is better than the 2022 d1280 (0.5594 against 0.5722)
with half its active parameters.

## 6. The fit, and the two targets

`scripts/old_vs_new/fit.py` reads every finished run under `runs/` whose name is a
ladder's, the searches' trials too (not a halving rung's, which trains fewer steps),
and takes each rung's best as its point. It fits each metric of each family as

$$L(N) = E + A N^{-\alpha},$$

N the active parameters besides the embeddings, with E shared: the families read the
same data, whose entropy neither can go below. It minimizes the Huber loss of log L,
as Hoffmann et al. (2022) did, and gives each prediction the range of the fits that
leave one rung out:

```console
$ python scripts/old_vs_new/fit.py runs --old 4096 --new 3072 --plot ladder.png
family  d_model  active   lr        objective   code_bpb   repo_bpb    web_bpb
old         512    12.6M  1.53e-03     0.8899     0.8051     0.7599     1.2971
old         768    42.5M  1.32e-03     0.7127     0.6253     0.5815     1.1281
old        1024   100.8M  2.26e-03     0.6268     0.5406     0.4976     1.0363
old        1280   196.8M  1.57e-03     0.5722     0.4891     0.4482     0.9659
old        1536   340.0M  1.43e-03     0.5345     0.4539     0.4138     0.9173
new         512    32.1M  2.31e-03     0.6715     0.5908     0.5253     1.0926
new         768   105.9M  1.24e-03     0.5594     0.4804     0.4249     0.9588
new        1024   248.1M  2.60e-03     0.5009     0.4256     0.3727     0.8814
new        1280   481.2M  2.60e-03     0.4597     0.3874     0.3370     0.8245
learning rates (vertex of a parabola in log lr, over each rung's trials):
  old d_model=512: 20 trials, best 0.00154
  old d_model=768: 18 trials, best 0.00158
  old d_model=1024: 9 trials, best 0.00179
  old: lr = 0.000479 N^0.070
  new d_model=512: 12 trials, best 0.0022
  new d_model=768: 3 trials, best 0.00161
  new: lr = 0.207 N^-0.263
objective: E = 0.3204
  old: A = 73.2, alpha = 0.297; at 6.44B: 0.4096 [0.3947, 0.4097]
  new: A = 99.2, alpha = 0.326; at 6.54B: 0.3823 [0.3785, 0.3862]
code_bpb: E = 0.2808
  old: A = 128, alpha = 0.336; at 6.44B: 0.3450 [0.3300, 0.3452]
  new: A = 196, alpha = 0.373; at 6.54B: 0.3234 [0.3195, 0.3265]
repo_bpb: E = 0.2448
  old: A = 130, alpha = 0.338; at 6.44B: 0.3072 [0.2821, 0.3104]
  new: A = 218, alpha = 0.385; at 6.54B: 0.2810 [0.2776, 0.2886]
web_bpb: E = 0.3832
  old: A = 13.1, alpha = 0.163; at 6.44B: 0.7135 [0.7075, 0.7181]
  new: A = 14.6, alpha = 0.175; at 6.54B: 0.6631 [0.6566, 0.6808]
plot: ladder.png
plot: ladder-dark.png
```

The script fits a learning-rate law to every family with two searched rungs; the 2026
one goes through the vertex of three trials 0.0007 apart, which is noise, and the
ladder does not use it. The 2022 prediction is at the top of its range: leaving any
rung out lowers it, so the fit of all five is the less flattering to the 2022 family.

```{image} old-vs-new/ladder.png
:alt: Each family's objective against its active parameters, its rungs, its fit, and a star at its target
:class: only-light
```

```{image} old-vs-new/ladder-dark.png
:alt: Each family's objective against its active parameters, its rungs, its fit, and a star at its target
:class: only-dark
```

The rungs are the dots, each family's fit the line, solid where it has rungs and dashed
where it extrapolates, and the stars the targets: the 2022 7B, 6.44B parameters a
token, and the 2026 model of `d_model = 3072`, 91B parameters, 6.54B of them a token.
Both would train on about 130B tokens, and serve agents as fast as each other.

## 7. What it says, and what it does not

At the same serving cost and the same training data, the ladders predict the 2026 model
the better one by every measure:

| | 2022, `d_model = 4096` | 2026, `d_model = 3072` | 2026 against 2022 |
|---|---|---|---|
| parameters (a token's) | 6.86B (6.44B) | 91.1B (6.54B) | |
| sessions an hour, 8 at once on a B300 | 892 | 894 | the same |
| objective | 0.410 [0.395, 0.410] | 0.382 [0.379, 0.386] | -6.7% |
| code files | 0.345 [0.330, 0.345] | 0.323 [0.320, 0.327] | -6.3% |
| repositories, rows of 16k | 0.307 [0.282, 0.310] | 0.281 [0.278, 0.289] | -8.5% |
| web text | 0.714 [0.708, 0.718] | 0.663 [0.657, 0.681] | -7.1% |

Bits per byte, each predicted by its family's fit, the leave-one-rung-out range in
brackets. The ranges part on every measure but the repositories, where they touch.

- **The gap narrows with scale, and stays.** By the fits, the 2026 family is 10%
  under the 2022 one at 100M active parameters, and 6.7% at the targets: both near
  the same entropy E, so the bits between them shrink, though the 2026 family's
  exponent is the larger (0.33 against 0.30). A token goes through as many
  parameters in both, but the 2026 model picks them from 13 times as many.
- **It gains most on long context.** On whole repositories in rows of 16k its edge is
  the largest, and it is not only its edge on code: every 2026 rung's bits per byte
  on repositories are 87 to 89% of its bits on code files, every 2022 rung's 91 to
  94%, so the 2026 rungs make more of the files before. Hybrid attention, RoPE on a
  quarter of its channels at a base of 10M, and the Gated DeltaNet's state are all
  built for it; the 2022 model has RoPE at 10k on every channel, which its long
  phase has to stretch. The repositories are Python's alone, and the comparison
  holds the data, not the length, fixed: an eval of the same repositories in rows of
  2k would part what the long rows add.
- **It trains more stably**: no learning rate up to 2.6e-2 diverged it, where the
  2022 family diverges above 7e-3.

What the experiment does not say:

- **The targets are extrapolations**, 19 times past the 2022 ladder's widest rung and
  14 times past the 2026 one's, from rungs of a single seed, 0.001 to 0.003 apart
  between two runs at the same point. The leave-one-out ranges are the fits'
  sensitivity to each rung, not a confidence interval.
- **The serving cost is one load on one GPU.** Under tensor parallelism, at other
  lengths of context or turns, or on other kernels (vLLM's Triton MoE kernels here,
  as FlashInfer's did not build), the 2026 model the same cost as the 7B would be
  another width. It also needs 182 GB of the GPU's 275 for its weights, which a
  smaller GPU does not have.
- **Each family is one point of a design space.** The 2026 family changes the
  attention, the mixer, the MoE, the norms and the positions at once; the ladders
  compare the two wholes, not what each change is worth. A sweep of one change at a
  time is the next experiment, and the same files run it.
- **Only the learning rate is tuned**, per family; the batch follows DeepSeek's law
  for both, and everything else is either family's own or the recipe's.
- **Twenty tokens a parameter is a short training.** Models of this size now see
  hundreds, and how each family's edge moves with the tokens a parameter, ladders all
  at 20 do not measure.

## What it takes

On one node of 8 B300s, each step as written:

| Step | what | time |
|------|------|------|
| 3 | the data: Stack v2 Edu converted, FineWeb-Edu downloaded, both packed | 10 min |
| 2 | the serving benchmark: six models, a GPU each, four loads | 22 min |
| 4 | the searches: 2022 d512 (20 trials), d768 (18), d1024 (8); 2026 d512 (12), d768 (9, halving) | 7 h 08 min |
| 5 | the ladders' rungs that no search trained: 2022 d1280 and d1536, 2026 d1024 and d1280 | 7 h 36 min |
| 6 | the fit | 1 min |

About 15 hours for all of it, a third of them the 2026 family's two widest rungs. The
searches' results are in the ladder files, so the ladders alone reproduce the fit, in
9 hours; without the 2026 d1280 rung, the widest and four of those hours, in under 5,
with a fit of three 2026 rungs.

## References

- Dehghani et al. (2023), *Scaling Vision Transformers to 22 Billion Parameters*,
  ICML, [arXiv:2302.05442](https://arxiv.org/abs/2302.05442): QK-norm.
- DeepSeek-AI (2024), *DeepSeek LLM: Scaling Open-Source Language Models with
  Longtermism*, [arXiv:2401.02954](https://arxiv.org/abs/2401.02954): the batch-size
  and learning-rate laws.
- DeepSeek-AI (2024), *DeepSeek-V3 Technical Report*,
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437): 256 fine-grained experts, 8
  routed and 1 shared.
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*,
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556): 20 tokens a parameter, and
  the fit.
- Kandpal et al. (2025), *The Common Pile v0.1*,
  [arXiv:2506.05209](https://arxiv.org/abs/2506.05209): Stack v2 Edu.
- Penedo et al. (2024), *The FineWeb Datasets: Decanting the Web for the Finest Text
  Data at Scale*, [arXiv:2406.17557](https://arxiv.org/abs/2406.17557): FineWeb-Edu.
- Qiu et al. (2025), *Gated Attention for Large Language Models*,
  [arXiv:2505.06708](https://arxiv.org/abs/2505.06708).
- Qwen Team (2025), *Qwen3-Next*: three Gated DeltaNet layers to one gated attention.
- Yang, Kautz & Hatamizadeh (2025), *Gated Delta Networks*, ICLR,
  [arXiv:2412.06464](https://arxiv.org/abs/2412.06464).
- Zhang et al. (2022), *OPT: Open Pre-trained Transformer Language Models*,
  [arXiv:2205.01068](https://arxiv.org/abs/2205.01068); Black et al. (2022),
  *GPT-NeoX-20B*, [arXiv:2204.06745](https://arxiv.org/abs/2204.06745): the 2022
  family.
