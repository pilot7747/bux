"""bux's pass over XLA's schedule (bux.train.reorder): a gradient's reduction starts as
soon as the gradient is computed, and is waited for before the next one starts; and the
registry of bux's passes it is in (bux.passes)."""

import pytest

from bux import jax_internals, passes
from bux.train import reorder

SUM = """
%add (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %s = f32[] add(%a, %b)
}
"""
SCATTER = """
%scatter{n} (x{n}: f32[8]) -> f32[1] {{
  %x{n} = f32[8]{{0}} parameter(0)
  ROOT %r{n} = f32[1]{{0}} reduce-scatter(%x{n}), replica_groups={{{{0,1,2,3,4,5,6,7}}}},
    dimensions={{0}}, to_apply=%add
}}
"""
# Three layers' backward passes, each a gradient by the layer's input (dx), which the
# next reads, and by its weights (grad), which only their reduction reads: XLA's
# scheduler puts each weight's gradient at the end, just before its reduction, each
# after the one before.
STEP = """
ENTRY %step (p: f32[8]) -> (f32[1], f32[1], f32[8], f32[8]) {
  %p = f32[8]{0} parameter(0)
  %dx1 = f32[8]{0} exponential(%p)
  %dx2 = f32[8]{0} exponential(%dx1)
  %dx3 = f32[8]{0} exponential(%dx2)
  %grad1 = f32[8]{0} sqrt(%dx1)
  %flat1 = f32[8]{0} bitcast(%grad1)
  %start1 = ((f32[8]{0}), f32[1]{0}) async-start(%flat1), calls=%scatter1
  %done1 = f32[1]{0} async-done(%start1)
  %grad2 = f32[8]{0} sqrt(%dx2)
  %start2 = ((f32[8]{0}), f32[1]{0}) async-start(%grad2), calls=%scatter2
  %done2 = f32[1]{0} async-done(%start2)
  %grad3 = f32[8]{0} sqrt(%dx3)
  %start3 = f32[8]{0} all-reduce-start(%grad3), replica_groups={{0,1,2,3,4,5,6,7}},
    to_apply=%add
  %done3 = f32[8]{0} all-reduce-done(%start3)
  ROOT %out = (f32[1]{0}, f32[1]{0}, f32[8]{0}, f32[8]{0}) tuple(%done1, %done2,
    %done3, %dx3)
}
"""


def scheduled(entry: str) -> bytes:
    scatters = "".join(SCATTER.format(n=n) for n in (1, 2))
    text = "HloModule step, is_scheduled=true\n" + SUM + scatters + entry
    return jax_internals.hlo_module_from_text(text).as_serialized_hlo_module_proto()


def order(serialized: bytes, computation: str = "step") -> list[str]:
    module = jax_internals.hlo_module(serialized)
    [found] = [c for c in module.make_nonfusion_computations() if c.name == computation]
    return [inst.name for inst in module.schedule().sequence(found)]


def test_each_weight_s_gradient_and_reduction_follow_its_layer_s_backward_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(reorder, "LAG", 1)
    moved = reorder.reordered(scheduled(STEP))
    assert moved is not None
    assert order(moved) == [
        *("p", "dx1", "grad1", "flat1", "start1"),
        *("dx2", "done1", "grad2", "start2"),  # waits for the one before, then starts
        *("dx3", "done2", "grad3", "start3"),
        *("done3", "out"),
    ]


NO_REDUCTION = """
ENTRY %step (p: f32[8]) -> f32[8] {
  %p = f32[8]{0} parameter(0)
  ROOT %grad = f32[8]{0} exponential(%p)
}
"""


def test_at_a_lag_of_two_a_reduction_is_waited_for_before_the_second_next():
    moved = reorder.reordered(scheduled(STEP))
    assert moved is not None
    assert reorder.LAG == 2
    assert order(moved) == [
        *("p", "dx1", "grad1", "flat1", "start1"),
        *("dx2", "grad2", "start2"),
        *("dx3", "done1", "grad3", "start3"),
        *("done2", "done3", "out"),
    ]


def test_a_schedule_whose_reductions_start_early_stays():
    once = reorder.reordered(scheduled(STEP))
    assert once is not None
    assert reorder.reordered(once) is None
    assert reorder.reordered(scheduled(NO_REDUCTION)) is None


def test_a_pass_that_fails_leaves_the_module_with_a_warning():
    def broken(serialized: bytes) -> bytes | None:
        raise ValueError("no such instruction")

    with pytest.warns(UserWarning, match="broken failed"):
        assert passes.guarded("broken", broken)(scheduled(STEP)) is None
    assert passes.PASSES[reorder.NAME].stage == passes.AFTER_SCHEDULING
