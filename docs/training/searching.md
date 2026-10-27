(training-searching)=
# Searching hyperparameters

A sweep (`a | b`, [Sweeps](../language/sweeps.md)) trains every combination of its
alternatives. A **search** trains a budget of **trials**, points of its
**dimensions**, and lets a method choose each point from the trials so far. It ranks
them by an **objective**, such as the bits per byte of held-out sets, weighted. A
setting written as a distribution is a dimension, and the file is a search:

```bux
optimizer.lr           = log_uniform(1e-4, 1e-2)
optimizer.weight_decay = uniform(0, 0.2)
objective              = 0.8 * web_bpb + 0.2 * code_bpb
search                 = search(method=gp, trials=27, halving=halving(eta=3, rungs=3))
```

This page covers dimensions, objectives, a search for each run of a sweep, the
methods and when to use each, halving, training a search and reading its trials, and
what the literature says about doing it well.

## A search in a file

`examples/experiments/attention_search.bux` compares three attentions, each at its own
best learning rate and weight decay. It is the attention ablation of
[Training](index.md#a-sweep) with its learning rates turned into a search:

```bux
use llama3_8b
use fineweb_wsd

d_model = 768
ffn.intermediate_size = 2048
model = embed block^12 norm head(tied=true)

attn.num_heads    = 12
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
budget            = 3B

# The dimensions of each search.
optimizer.lr           = log_uniform(1e-4, 1e-2)
optimizer.weight_decay = uniform(0, 0.2)

# Held-out web text and code, and what the search minimizes: their bits per byte,
# weighted as the data is.
data = 0.8 * web(split=[0, 0.999]) + 0.2 * code(split=[0, 0.999])
eval = [eval(name="web", data=web(split=[0.999, 1]), every=1k),
        eval(name="code", data=code(split=[0.999, 1]), every=1k)]
objective = 0.8 * web_bpb + 0.2 * code_bpb

search = search(method=gp, trials=27, halving=halving(eta=3, rungs=3))
# A short run is the long one shrunk: its warmup too, a share of its steps.
schedule.warmup = 0.15
```

`bux check` says what it searches, what it minimizes and what halving trains:

```console
$ bux check examples/experiments/attention_search.bux | tail -4
search: gp, 27 trials over optimizer.lr (log_uniform(1e-4, 0.01)) × optimizer.weight_decay (uniform(0, 0.2))
objective = 0.8 * web_bpb + 0.2 * code_bpb
halving: rungs of 636, 1.91k, 5.72k steps, eta 3, about 9 runs' steps in all
sweep: 3 runs over attn.num_kv_heads (mha | gqa | mqa)
```

The `|` makes three runs, and each is a search of its own: 27 trials of MHA, 27 of
GQA, 27 of MQA. With halving, every trial trains 636 steps, the best third of those
1.91k, and the best third of those the run's 5.72k, so each search costs about nine
full runs, not 27. `bux plan` lists the three runs, each dimension at its middle, and
a line for each search:

```console
$ bux plan examples/experiments/attention_search.bux
3 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
27b626130f  183M    3B      16.4       1.61B      1     700MiB      1.37GiB    0%      attention_search[mha]
b16db24d21  174M    3B      17.2       1.55B      1     664MiB      1.3GiB     0%      attention_search[gqa]
3a7f26f452  170M    3B      17.6       1.53B      1     650MiB      1.27GiB    0%      attention_search[mqa]
attention_search[mha]: 27 trials of 2 dimensions by gp, 3 rungs: about 9 runs' steps
attention_search[gqa]: 27 trials of 2 dimensions by gp, 3 rungs: about 9 runs' steps
attention_search[mqa]: 27 trials of 2 dimensions by gp, 3 rungs: about 9 runs' steps
```

`bux launch` trains the trials side by side, a GPU group each, and `bux results`
prints each search's trials, the best first ([Training a search](#training-a-search)).

## Dimensions

A dimension is a setting written as a distribution, anywhere a number goes, or, for
`choice`, anywhere a value or a block goes:

| Written | What a trial takes |
|---------|--------------------|
| `uniform(0, 0.2)` | a number between the two, uniformly |
| `log_uniform(1e-4, 1e-2)` | a number between the two, uniformly on a log scale |
| `int_uniform(100, 2k)` | an integer from the first to the second, both included; `log=true` on a log scale |
| `choice(swish, gelu)` | one of the options: any values, settings or blocks |
| `choice(pre=[norm attn] [norm ffn], par=[norm (attn + ffn)])` | one of the options, named |

The bounds come in order, `low` then `high`, or by name: `uniform(low=0, high=0.2)`.
Each is a config, so the [language reference](../reference/language/index.md) and the
editor's hover document it.

- **Outside a search, a dimension takes its middle.** `bux check`, `bux plan`, `bux
  fmt` and `compile_runs` show the run halfway along each scale: the geometric mean
  of a `log_uniform`, the first option of a `choice`.
- **A trial's values have three significant digits**, so its run is named for
  exactly what it trains, and its `run.bux` compiles back to the same run:
  `attention_search[mha, optimizer.lr=3.16e-4, optimizer.weight_decay=0.0712]`.
  A choice names it by its option's name, `block=par`, or as written, `ffn.activation=gelu`.
  Two trials that take the same values, as trials of a few options must, are one
  run, trained once.
- **A dimension is named by where it is written**, as an unlabeled alternative is:
  `optimizer.lr`, or the name of a definition, `lr = log_uniform(1e-4, 1e-2)`,
  which other settings can then share: `optimizer = muon(lr=20 * lr, adam=adamw(lr=lr))`.
  Inside a macro, it is the macro's name, and it takes the same value at every call.
- **A dimension inside an option of a choice** is a dimension of the search too,
  which only the trials that take that option reach:
  `optimizer = choice(adamw=adamw(lr=log_uniform(1e-4, 1e-2)), muon=muon(lr=log_uniform(1e-3, 0.1)))`.
- **`bux check` tries each dimension at both ends**, and each option of a choice,
  with the others at their middles, as it tries every alternative of a sweep:

  ```console
  $ bux check examples/experiments/smoke.bux 'attn.num_kv_heads = int_uniform(1, 3)'
  error: gqa: num_kv_heads: Value error, 3 key and value heads do not divide 4 query heads
   --> <command line>:1:6
    |
  1 | attn.num_kv_heads = int_uniform(1, 3)
    |      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ```

  A setting that only some integers satisfy, as a number of heads, is a `choice` of
  them: `choice(1, 2, 4)`.

## The objective

A search minimizes the run's `objective`: a weighted sum of the metrics it logs, each
at its latest value, which for an eval is the one after the last step.

```bux
objective = 0.8 * web_bpb + 0.2 * code_bpb   # held-out bits per byte, weighted
objective = (web_bpb + code_bpb) / 2         # their mean
objective = eval_loss                         # one eval's loss
objective = -tokens_per_s                     # the fastest, to tune `methods`, say
```

- **Its names are metrics, not definitions**, but for a definition of a number:
  with `w = 0.8`, `w * web_bpb + (1 - w) * code_bpb` is the first line. A metric
  with a slash is a string: `"time/step_ms"`.
- **An eval's metrics are checked against the run's evals**: `web_bpb` needs an eval
  named `web`, and bits per byte need a tokenizer ([Eval](running.md#eval)).
- **Without one, the objective is the first eval's bits per byte** if the run has a
  tokenizer, else that eval's loss, else the training loss.
- A trial that failed, or whose objective is not a number, as a run that diverged
  has, counts as failed: the methods learn to avoid where it was.

A sweep that sets an `objective` shows it in the table `bux launch` ends with, so the
same line ranks the runs of a grid.

Neither `objective` nor `search` is in a run's id ([Run ids](running.md#run-ids)): a
trial that a sweep, or another search, trained already is read, not trained again.

## A search for each run of a sweep

The alternatives of a sweep stay a grid, and each run of it is a search of its own,
over the dimensions it reaches. So `attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1`
with `optimizer.lr = log_uniform(1e-4, 1e-2)` compares three attentions, each at its
best learning rate, which is the fair comparison when the best learning rate moves
with the architecture: comparisons against a baseline tuned less carefully overstate
the differences (Wen et al., 2025, for optimizers). The searches take turns at the
GPUs.

To let the search choose the attention too, write it as a dimension:
`attn.num_kv_heads = choice(12, 4, 1)`.

## Methods

`search.method` proposes the next trial from the trials so far. Each is found by
name in the `bux.search.methods.methods` registry
([Registries](../reference/language/registries.md)):

| Method | What it does | For |
|--------|--------------|-----|
| `gp` (the default) | Bayesian optimization: a Gaussian process with a Matérn-5/2 kernel, a length scale per dimension, fit to the trials; the next point has the most expected improvement (log EI) | few trials of long runs, in a few dimensions |
| `tpe` | the tree-structured Parzen estimator, as Optuna 5's: densities of the best trials and of the rest; the next point is likeliest under the first and least under the second | many choices, dimensions inside choices, or more than about 50 trials |
| `halton` | quasi-random: a scrambled Halton sequence, even coverage for any number of trials | exploring a space, or a baseline |
| `sobol` | quasi-random: a scrambled Sobol sequence, evenest when the trials are a power of two | the same, with 8, 16 or 32 trials |
| `random` | uniformly random points | a baseline |

`gp` and `tpe` start with quasi-random trials (`startup`: 5 and 10), then fit their
model to every trial that has an objective. A trial still training, or with halving
one trained only at a rung below the one they model, counts too: `gp` at its
prediction for it (the kriging believer) and `tpe` among the worse (the constant
liar), so trials that start together differ.

Which to use: the Deep Learning Tuning Playbook (Godbole et al., 2023) explores a new
space with quasi-random search, which needs nothing of the trials' results and shows
where the good region is, and turns to Bayesian optimization to exploit it: "if you
can only run a few trials in parallel but many in sequence, Bayesian optimization
becomes much more attractive". With 10 to 50 trials of a few dimensions, a GP is the
most sample-efficient of these; TPE wins with many choices or many trials.

## Halving

`halving` trains every trial short first, and only the best longer: asynchronous
successive halving, ASHA (Li et al., 2020). With `halving(eta=3, rungs=3)`, rung 0 is
a ninth of the run's steps, rung 1 a third, rung 2 all of them. A trial that finished
a rung goes up one when it is among the best third of those that finished it. A GPU
that frees up takes the next trial up, the highest rung first, and if none has
earned it, a new trial at the bottom: no rung waits for the rest of it.

- **Each rung is a run of its own, from scratch**, with its schedule over its own
  steps: a short run's cooldown is the long run's, shrunk, and so its objective is the
  best that length reaches. Write what depends on the run's length as a share of
  it, as `schedule.warmup = 0.15`, or in terms of `steps`; `bux check` compiles the
  bottom rung, and says so if a warmup does not fit in it.
- **`trials` start at the bottom**, and about `trials / eta^k` reach rung `k`: 27
  trials with three rungs of `eta=3` are 27 short runs, 9 middle ones and 3 full ones,
  about `rungs × trials / eta^(rungs - 1)` = 9 runs' steps in all.
- **The method models one rung**: the highest at which at least one trial more than
  there are dimensions has an objective, as BOHB does (Falkner et al., 2018).
- **The best trial** is the least at the highest rung reached.
- **`brackets`** above 1 is Hyperband (Li et al., 2018): trials take turns starting
  at rung 0, 1, ..., `brackets - 1`, and each bracket ranks its own, which hedges
  against short runs that rank trials as long ones would not.

A short run favors a high learning rate: the best learning rate falls as a run gets
longer (Bjorck et al., 2025). Halving keeps the choice at the full length, where the
best trial is ranked, but it can drop a good low learning rate at the bottom rung.
Hyperband's authors recommend an `eta` of 3 or 4.

## Training a search

| | `bux train` | `bux launch` |
|---|---|---|
| Trials | one after another, in this process | side by side, a worker each, on the GPUs its mesh needs |
| The next trial | proposed as one ends | proposed as GPUs free up, from those done and those training |
| A trial that is done | read, not trained again | read, not trained again |
| SIGTERM, Ctrl-C | stops the trial after its step, saving it with a `checkpoint` setting, and the search | the same, in every worker |

Each trial's rung is a run, with its own directory, `status.json` and metrics, as any
run of a sweep ([Launching](launching.md)). `search.parallel` caps the trials one
search trains at once; by default, as many as the GPUs hold. A search ends with its
table, the best first, and `bux results FILE` prints it again at any time, without
training. Settings on the command line change it as they change a run:
`bux launch tune.bux 'search.trials = 32'`.

A search can come from the command line, as any setting can. This one tunes the
learning rate of `smoke.bux` on a laptop's CPU, by the default `gp`, with two rungs:
17 steps, then the run's 50. The first trials are quasi-random, and each trains as a
run of its own:

```console
$ bux train examples/experiments/smoke.bux 'optimizer.lr = log_uniform(1e-3, 1e-1)' 'search = search(trials=9, halving=halving(rungs=2))'
smoke[optimizer.lr=0.026, steps=17]: training on cpu of my-laptop with jax 0.12.0
smoke[optimizer.lr=0.026, steps=17]: runs/smoke_optimizer.lr=0.026_steps=17-bb7efd8065, on 1 × cpu
smoke[optimizer.lr=0.026, steps=17]: 156k parameters, 17 steps × 8 × 64 tokens = 8.7k tokens, adamw(lr=0.026), wsd(warmup=10)
smoke[optimizer.lr=0.026, steps=17]: log every 10
remat=auto: none in 0s (none 0.0 GiB in 0s)
smoke[optimizer.lr=0.026, steps=17] step 10/17  loss 6.3674  lr 2.60e-02  grad 0.453  7,977 tok/s  64 ms/step  data 1.4 ms  compiled in 0.5s  0:00:01 < 0:00:00
smoke[optimizer.lr=0.026, steps=17] step 17/17  loss 6.4086  lr 0.00e+00  grad 0.394  70,488 tok/s  7 ms/step  data 0.0 ms  0:00:01 < 0:00:00
smoke[optimizer.lr=0.026, steps=17]: done
...
smoke[optimizer.lr=0.00109]: runs/smoke_optimizer.lr=0.00109-2974712d5c, on 1 × cpu
smoke[optimizer.lr=0.00109]: 156k parameters, 50 steps × 8 × 64 tokens = 25.6k tokens, adamw(lr=0.00109), wsd(warmup=10)
...
search smoke: 9 of 9 trials; objective = loss; best 6.2600, trial 8
trial  rung  objective  run
8      1/1   6.2600     smoke[optimizer.lr=0.00109]
6      1/1   6.2625     smoke[optimizer.lr=0.00233]
1      1/1   6.2625     smoke[optimizer.lr=0.0026]
7      1/1   6.2633     smoke[optimizer.lr=0.0017]
5      1/1   6.2677     smoke[optimizer.lr=0.00385]
3      0/1   6.2786     smoke[optimizer.lr=0.00821, steps=17]
4      0/1   6.2964     smoke[optimizer.lr=0.0146, steps=17]
0      0/1   6.4086     smoke[optimizer.lr=0.026, steps=17]
2      0/1   6.4199     smoke[optimizer.lr=0.0821, steps=17]
```

Nine trials trained 17 steps, and the five that earned it the full 50: 14 runs, a
few seconds each. The table lists each trial at the highest rung it reached, the best
first. The smoke run has no eval, so its objective is the training loss. On random
tokens every loss tends to log 512 ≈ 6.24, near where a run at bux's default init
starts, and the best learning rates are the ones that stray from it least; trial 8
sits near the dimension's lower end, so on real data the next step would be to widen
it ([Doing it well](#doing-it-well)).

## The search directory

A search keeps its trials in `runs/<name>-search-<key>/`, beside their runs:

```text
trials.jsonl   a line for each trial proposed, its values and bracket, and one for each rung it starts
best.bux       the run.bux of the best trial so far, at the highest rung
```

The key hashes what the search is over and how: its dimensions, `method`, `halving`
and `seed`. So:

- **A search launched again goes on where it stopped.** Its done trials are read, a
  trial that a signal stopped trains again, from its checkpoint if it has a
  `checkpoint` setting, and the method proposes the rest from all of them.
- **More `trials` add trials** to the same search.
- **Another space, method or seed is a search of its own**, in a directory of its
  own; a trial at a point another search trained already is read from its run,
  and one at a point another search is training waits for it.
- **A trial that failed stays failed**, as information: its point is one to avoid.
  Delete its run's directory to try it again.
- **The best trial is a file**: `bux train runs/<name>-search-<key>/best.bux` trains
  it again, and `'seed = 1 | 2 | 3'` after it, with other seeds.

## Doing it well

What the literature agrees on, for searches of expensive runs:

- **Search a learning rate, a weight decay, a warmup, an epsilon on a log scale.**
  Their good values span orders of magnitude.
- **A best trial at the edge of a dimension** means the space is too small: widen it
  and launch again. The trials already trained stay where they are.
- **Every trial has the run's `seed`**, so they see the same data in the same order
  and start from the same initialization: their differences are the dimensions'.
  Measure the noise between seeds once, and train the best trial with fresh seeds
  before trusting a small difference (the Tuning Playbook; CARBS samples its best
  points again for the same reason).
- **Compare at the length you care about.** An objective of a short run, or of a run
  midway through its schedule, ranks learning rates as a long run would not (Wu et
  al., 2018; Hägele et al., 2024).
- **Tune small, transfer large**: hyperparameters found on a small model transfer to
  wider and deeper ones under µP and its successors (Yang et al., 2022; Everett et
  al., 2024), and the best learning rate and batch follow power laws of compute and
  tokens (DeepSeek-AI, 2024; Li et al., 2025).

## A method of your own

A method is a function registered under the name of its config's type, which gets the
config, the space of coordinates (`bux.search.space`) and the history so far, and
returns the next point. Registered under a taken name, it replaces bux's own:

```python
from bux.search.methods import History, Point, generator, methods
from bux.search.space import Space
from bux.train.config import RandomSearch


@methods.register("random")
def _(c: RandomSearch, space: Space, history: History) -> Point:
    """Random points, but never two alike in their first dimension."""
    rng = generator(history)
    point = rng.random(len(space.names))
    while any(abs(point[0] - p[0]) < 0.05 for p in history.points):
        point = rng.random(len(space.names))
    return space.from_unit(point)
```

A number's coordinate is in [0, 1], the fraction of the way along its scale; a
choice's is its option's index. A new name needs its config in the `SearchMethod`
union of `bux.train.config` too ([Extending the trainer](extending.md)).

## References

- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR.
- Bergstra, Bardenet, Bengio & Kégl (2011), *Algorithms for Hyper-Parameter
  Optimization*, NeurIPS: TPE.
- Watanabe (2023), *Tree-Structured Parzen Estimator: Understanding Its Algorithm
  Components and Their Roles for Better Empirical Performance*,
  [arXiv:2304.11127](https://arxiv.org/abs/2304.11127).
- Snoek, Larochelle & Adams (2012), *Practical Bayesian Optimization of Machine
  Learning Algorithms*, NeurIPS.
- Hvarfner, Hellsten & Nardi (2024), *Vanilla Bayesian Optimization Performs Great in
  High Dimensions*, ICML, [arXiv:2402.02229](https://arxiv.org/abs/2402.02229).
- Ament, Daulton, Eriksson, Balandat & Bakshy (2023), *Unexpected Improvements to
  Expected Improvement for Bayesian Optimization*, NeurIPS,
  [arXiv:2310.20708](https://arxiv.org/abs/2310.20708): log EI.
- Ginsbourger, Le Riche & Carraro (2010), *Kriging Is Well-Suited to Parallelize
  Optimization*: the kriging believer and the constant liar.
- Song et al. (2024), *The Vizier Gaussian Process Bandit Algorithm*,
  [arXiv:2408.11527](https://arxiv.org/abs/2408.11527).
- Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2018), *Hyperband: A Novel
  Bandit-Based Approach to Hyperparameter Optimization*, JMLR.
- Li et al. (2020), *A System for Massively Parallel Hyperparameter Tuning*, MLSys,
  [arXiv:1810.05934](https://arxiv.org/abs/1810.05934): ASHA.
- Falkner, Klein & Hutter (2018), *BOHB: Robust and Efficient Hyperparameter
  Optimization at Scale*, ICML.
- Godbole, Dahl, Gilmer, Shallue & Nado (2023), *Deep Learning Tuning Playbook*,
  [github.com/google-research/tuning_playbook](https://github.com/google-research/tuning_playbook).
- Owen (2020), *On dropping the first Sobol' point*,
  [arXiv:2008.08051](https://arxiv.org/abs/2008.08051).
- Wu, Ren, Liao & Grosse (2018), *Understanding Short-Horizon Bias in Stochastic
  Meta-Optimization*, ICLR.
- Hägele et al. (2024), *Scaling Laws and Compute-Optimal Training Beyond Fixed
  Training Durations*, NeurIPS, [arXiv:2405.18392](https://arxiv.org/abs/2405.18392).
- Bjorck, Benhaim, Chaudhary, Wei & Song (2025), *Scaling Optimal LR Across Token
  Horizons*, ICLR, [arXiv:2409.19913](https://arxiv.org/abs/2409.19913).
- Wen et al. (2025), *Fantastic Pretraining Optimizers and Where to Find Them*,
  [arXiv:2509.02046](https://arxiv.org/abs/2509.02046).
- Yang et al. (2022), *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
  Hyperparameter Transfer*, [arXiv:2203.03466](https://arxiv.org/abs/2203.03466): µP.
- Everett et al. (2024), *Scaling Exponents Across Parameterizations and Optimizers*,
  ICML, [arXiv:2407.05872](https://arxiv.org/abs/2407.05872).
- DeepSeek-AI (2024), *DeepSeek LLM: Scaling Open-Source Language Models with
  Longtermism*, [arXiv:2401.02954](https://arxiv.org/abs/2401.02954).
- Li et al. (2025), *Predictable Scale: Part I — Optimal Hyperparameter Scaling Law
  in Large Language Model Pretraining*,
  [arXiv:2503.04715](https://arxiv.org/abs/2503.04715): the Step Law.
- Fetterman et al. (2023), *Tune As You Scale: Hyperparameter Optimization For
  Compute Efficient Training*, [arXiv:2306.08055](https://arxiv.org/abs/2306.08055):
  CARBS.
