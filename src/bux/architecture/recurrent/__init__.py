"""Token mixers with a recurrent state: linear attention (``linear_attention``,
``rwkv7``) and state space models (``ssm``). Their recurrence runs by chunks
(``chunkwise``), the state carried across them by ``affine_scan``; the delta rule's
also runs by cuDNN's fused kernels (``fused_delta``), and with a decay per channel by
bux's own (``pallas_delta``).

Each recurrence starts from a zero state at every document's start, so a packed row
gives each document what it gives alone, and runs by the run's
``methods.recurrence`` (`bux.parallel.recurrence`): ``scan``, the default, or
``pass_state``, which splits the positions over ``context``. ``ssm`` also holds
LFM2's gated short convolution, ``short_conv``.
"""
