"""Token mixers with a recurrent state: linear attention (``linear_attention``,
``rwkv7``) and state space models (``ssm``). Their recurrence runs by chunks
(``chunkwise``), the state carried across them by ``affine_scan``; the delta rule's
also runs by cuDNN's fused kernels (``fused_delta``), and with a decay per channel by
bux's own (``pallas_delta``)."""
