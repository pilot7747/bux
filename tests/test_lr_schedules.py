"""Learning-rate schedules (bux.train.optim.schedules): each curve at every step of a
short run, against the formula its docs give."""

import itertools
import math

import pytest
from pydantic import ValidationError

from bux.train import config as tcfg
from bux.train.optim import lr_multiplier


def values(c: tcfg.ScheduleConfig, steps: int, **train: object) -> list[float]:
    """The multiplier of `c` at each step of a run of `steps`, to 4 places."""
    settings: dict[str, object] = {"data": tcfg.SyntheticConfig()}
    settings["optimizer"] = tcfg.AdamWConfig(lr=1)
    settings |= {"seq_len": 2, "batch": 1, "schedule": c, "steps": steps, **train}
    f = lr_multiplier(tcfg.TrainConfig.model_validate(settings))
    return [round(float(f(i)), 4) for i in range(steps)]


def test_schedules_warm_up_and_reach_final_at_the_last_step():
    assert values(tcfg.ConstantSchedule(warmup=4), 6) == [0.25, 0.5, 0.75, 1, 1, 1]
    cosine = values(tcfg.CosineSchedule(warmup=2, final=0.1), 6)
    assert cosine[:2] == [0.5, 1.0] and cosine[-1] == 0.1
    assert all(a >= b for a, b in itertools.pairwise(cosine[1:]))  # after warmup
    wsd = values(tcfg.WSDSchedule(decay=0.4, shape="linear"), 10)
    assert wsd == [1, 1, 1, 1, 1, 1, 0.75, 0.5, 0.25, 0]
    sqrt = values(tcfg.WSDSchedule(decay=0.4, shape="sqrt", final=0.1), 10)
    assert sqrt[5] == 1 and sqrt[6] == round(0.1 + 0.9 * (1 - 0.25**0.5), 4)


def test_a_warmup_below_1_is_a_share_of_the_steps():
    assert values(tcfg.ConstantSchedule(warmup=0.5), 4) == [0.5, 1, 1, 1]
    assert values(tcfg.LinearDecaySchedule(warmup=0.25), 8)[:3] == [0.5, 1, 0.8333]


def test_a_cosine_past_its_horizon_is_cut_off_and_one_before_it_holds():
    longer = values(tcfg.CosineSchedule(final=0, horizon=2), 4)
    assert longer == [
        round(0.5 * (1 + math.cos(math.pi * s / 8)), 4) for s in (1, 2, 3, 4)
    ]
    assert values(tcfg.CosineSchedule(final=0.2, horizon=0.5), 4) == [
        0.6,
        0.2,
        0.2,
        0.2,
    ]


def test_cooldowns_take_every_shape_to_final():
    def cooldown(shape: tcfg.Shape) -> list[float]:
        return values(tcfg.WSDSchedule(decay=0.5, final=0.1, shape=shape), 8)[3:]

    e = math.exp(-1)
    exp = [0.1 + 0.9 * (math.exp(-t / 4) - e) / (1 - e) for t in (1, 2, 3, 4)]
    assert cooldown("exp") == [1, *(round(m, 4) for m in exp)]
    assert cooldown("geometric") == [
        1,
        *(round(0.1 ** (t / 4), 4) for t in (1, 2, 3, 4)),
    ]
    assert cooldown("constant") == [1, 0.1, 0.1, 0.1, 0.1]


def test_linear_decay_goes_straight_from_the_warmup_to_final():
    assert values(tcfg.LinearDecaySchedule(warmup=2), 6) == [0.5, 1, 0.75, 0.5, 0.25, 0]
    assert values(tcfg.LinearDecaySchedule(final=0.5), 3)[-1] == 0.5


def test_multistep_drops_by_its_factor_at_each_share():
    c = tcfg.MultiStepSchedule(at=(0.5, 0.8), factor=0.5)
    assert values(c, 10) == [1] * 5 + [0.5] * 3 + [0.25] * 2


def test_power_follows_the_batch_and_tokens_then_cools_down():
    """a · batch · n^b over the peak, capped at 1, held through the warmup at its last
    step's, then IBM's exponential cooldown from there to 0."""
    c = tcfg.PowerSchedule(warmup=2, a=1, b=-0.5, decay=0.25)
    found = values(c, 20, batch=2, seq_len=4, optimizer=tcfg.AdamWConfig(lr=0.25))

    def power(s: int) -> float:  # 1 · 2 · (8 (s + 1))^-0.5 / 0.25
        return min(1.0, 8 / math.sqrt(8 * (s + 1)))

    e = math.exp(-1)
    cool = [(math.exp(-t / 5) - e) / (1 - e) for t in range(1, 6)]
    expected = [0.5 * power(1), power(1), *map(power, range(2, 15))]
    expected += [power(14) * f for f in cool]
    assert found == [round(m, 4) for m in expected]
    assert found[6] == 1 and found[8] < 1 and found[-1] == 0


def test_piecewise_holds_decays_and_drops_from_piece_to_piece():
    """DeepSeek-V3's shape, small: the peak, a cosine to 0.1, then 0.05 at once."""
    c = tcfg.PiecewiseSchedule(
        warmup=2,
        pieces=(
            tcfg.Piece(until=0.4),
            tcfg.Piece(until=0.8, to=0.1, shape="cosine"),
            tcfg.Piece(to=0.05),
        ),
    )
    cosine = [0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / 4)) for t in (1, 2, 3, 4)]
    assert values(c, 10) == [0.5, 1, 1, 1, *(round(m, 4) for m in cosine), 0.05, 0.05]


def test_a_piece_can_warm_up_again():
    pieces = (tcfg.Piece(until=0.5, to=0.2), tcfg.Piece(to=0.6, shape="linear"))
    found = values(tcfg.PiecewiseSchedule(pieces=pieces), 8)
    assert found == [0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6]


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: tcfg.Piece(until=1.5), "less than or equal to 1"),
        (
            lambda: tcfg.PiecewiseSchedule(pieces=(tcfg.Piece(until=0.5),)),
            "the last piece ends at 1",
        ),
        (
            lambda: tcfg.PiecewiseSchedule(
                pieces=(tcfg.Piece(until=0.6), tcfg.Piece(until=0.4), tcfg.Piece())
            ),
            "the shares of the steps must increase",
        ),
        (
            lambda: tcfg.PiecewiseSchedule(
                pieces=(
                    tcfg.Piece(until=0.5, to=0),
                    tcfg.Piece(to=0.1, shape="geometric"),
                )
            ),
            "starts and ends above 0",
        ),
        (lambda: tcfg.WSDSchedule(shape="geometric"), "starts and ends above 0"),
        (lambda: tcfg.MultiStepSchedule(at=(0.9, 0.8)), "must increase"),
    ],
)
def test_schedules_that_cannot_run_are_errors(make, message):
    with pytest.raises(ValidationError, match=message):
        make()
