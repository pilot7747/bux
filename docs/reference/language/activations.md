# Activations

An activation name is a layer that applies it elementwise, `linear(features=h) swish`,
and a value for settings that take one, `gated_mlp(activation=swish)`.

```{include} ../../_generated/activations.md
```

`situ` is `4 * tanh(x / 4) * sigmoid(x)`: SiLU with its linear factor soft-capped
(Kimi K3). `relu2` is the square of ReLU.
