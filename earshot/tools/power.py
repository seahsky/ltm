"""What effect size the matrix can actually detect, at a given number of episodes.

    python -m earshot.tools.power                       # the table, at the defaults
    python -m earshot.tools.power --n-per-cell 200
    python -m earshot.tools.power --target 0.10         # episodes needed for a 10-point MDE

Stdlib only, no GPU, instant. `statistics.NormalDist` supplies the quantiles.

**Why this is a module and not a calculation in a message.** ADR-0018 spent three amendments
arguing about power from figures nobody could re-derive, and the first table put in front of a
decision used **2 sigma** and called it an MDE. Two sigma is the significance threshold: an
effect sitting exactly there is detected half the time. The conventional MDE at 80% power is
2.80 sigma, so that table understated every requirement by about 40%. A number that decides how
many GPU-hours to spend belongs somewhere it can be recomputed and checked.

**Two different comparisons, two different formulas, and mixing them is the trap.**

- *Between cells* of the 2x2 (seen-heard against unseen-unheard, say) is an UNPAIRED comparison
  of two proportions: different scenes, different classes, different episodes. Nothing pairs.
- *Between arms* of the same sweep re-run (`yield-2` against `repeat-1`) is PAIRED per episode,
  which is what `episode_diff` exploits and where the measured 16.2% flip rate gives
  `SD = sqrt(f*n)` episodes. That is the formula behind "MDE = 15 episodes = 4.1 points at
  n=365", and it does NOT apply to the matrix.

Carrying the paired number into an unpaired design claims about 1.8x the sensitivity the
design has: at the same total rendering cost, 5.9 points paired against 10.4 unpaired. That
ratio was itself asserted as "roughly three times" before anything computed it, which is the
same class of error as the 2-sigma one and is why both are now under test.

**Episodes are not the binding constraint; SCENES are.** Episodes inside a scene share a room,
a source and a renderer, so they are not independent. `funnel_diff`'s scene-level sign test
already disagreed with an episode-level McNemar once, and the rule that came out of it is to
report both. `sign_test_threshold` prices the scene-level test, and at ten scenes a side the
answer is bleak in a way no episode count repairs.
"""

from __future__ import annotations

import argparse
import math
from statistics import NormalDist
from typing import Optional, Sequence

__all__ = [
    "sd_between_cells",
    "mde_between_cells",
    "episodes_for_mde",
    "sd_paired",
    "mde_paired",
    "sign_test_threshold",
    "main",
]

# provenance: measured -- the per-episode outcome flip rate on byte-identical re-runs
# (`repeat-1` against `arrive-2`, 2026-08-11). It prices the PAIRED comparison only.
MEASURED_FLIP_RATE = 0.162

# provenance: measured -- the memory effect this project has previously seen (+0.171 SR, the M3
# revisit headline). The matrix is being sized to detect something of that order.
HISTORICAL_EFFECT = 0.171


def _z(alpha: float, power: float):
    """`(z_alpha_two_sided, z_beta)`. Raises on a probability that is not one."""
    for name, value in (("alpha", alpha), ("power", power)):
        if not 0.0 < value < 1.0:
            raise ValueError("{} must be strictly between 0 and 1, got {}".format(name, value))
    normal = NormalDist()
    return normal.inv_cdf(1.0 - alpha / 2.0), normal.inv_cdf(power)


def sd_between_cells(n_per_cell: int, p: float = 0.5) -> float:
    """Standard error of the DIFFERENCE between two independent cells' success rates.

    `p = 0.5` is the worst case and the honest default: it maximises the variance, so an MDE
    computed at it cannot be beaten by a lucky base rate.
    """
    if n_per_cell < 1:
        raise ValueError("n_per_cell must be at least 1, got {}".format(n_per_cell))
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability, got {}".format(p))
    return math.sqrt(2.0 * p * (1.0 - p) / n_per_cell)


def mde_between_cells(
    n_per_cell: int, p: float = 0.5, alpha: float = 0.05, power: float = 0.8
) -> float:
    """The smallest between-cell difference detectable at `power`, as a fraction.

    NOT 2 sigma. At 80% power and a two-sided 0.05 the multiplier is 2.80, and the difference
    between the two is what made a 6-hour run look adequate for an effect it would miss one
    time in two.
    """
    z_alpha, z_beta = _z(alpha, power)
    return (z_alpha + z_beta) * sd_between_cells(n_per_cell, p)


def episodes_for_mde(
    target: float, p: float = 0.5, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Episodes per cell needed to detect `target`. Rounded UP; a partial episode is none."""
    if not 0.0 < target < 1.0:
        raise ValueError("target must be strictly between 0 and 1, got {}".format(target))
    z_alpha, z_beta = _z(alpha, power)
    return int(math.ceil(2.0 * p * (1.0 - p) * ((z_alpha + z_beta) / target) ** 2))


def sd_paired(n: int, flip_rate: float = MEASURED_FLIP_RATE) -> float:
    """SD of the difference in EPISODE COUNT between two paired arms, in episodes.

    Under the null the discordant pairs split evenly, so `Var(b - c) = b + c = flip_rate * n`.
    This is the number behind "SD(difference) 7.7" at n=365, and it belongs to `episode_diff`,
    not to the matrix.
    """
    if n < 1:
        raise ValueError("n must be at least 1, got {}".format(n))
    if not 0.0 <= flip_rate <= 1.0:
        raise ValueError("flip_rate must be a probability, got {}".format(flip_rate))
    return math.sqrt(flip_rate * n)


def mde_paired(
    n: int,
    flip_rate: float = MEASURED_FLIP_RATE,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """The smallest paired difference detectable, as a FRACTION of `n`."""
    z_alpha, z_beta = _z(alpha, power)
    return (z_alpha + z_beta) * sd_paired(n, flip_rate) / n


def sign_test_threshold(n_scenes: int, alpha: float = 0.05) -> Optional[int]:
    """How many of `n_scenes` must move the same way for a two-sided sign test to reach `alpha`.

    `None` means NO outcome reaches it: with too few scenes even a clean sweep is not
    significant. That is the case the matrix is in at ten scenes a side, and it is a property
    of the scene count that no number of episodes can repair.
    """
    if n_scenes < 1:
        raise ValueError("n_scenes must be at least 1, got {}".format(n_scenes))
    for k in range(n_scenes, n_scenes // 2, -1):
        tail = sum(math.comb(n_scenes, i) for i in range(k, n_scenes + 1))
        if 2.0 * tail / (2.0**n_scenes) > alpha:
            return k + 1 if k < n_scenes else None
    return n_scenes // 2 + 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Effect sizes the matrix can detect. No GPU.")
    parser.add_argument("--n-per-cell", type=int, default=None)
    parser.add_argument("--target", type=float, default=None)
    # A PAIRED sweep -- `ablation_sweep.sh`, or any two arms `episode_diff` can pair --
    # must be priced by the paired formula, and the paired block below was fixed at the
    # n=365 of the run that measured the flip rate. Passing the sweep's own n is how a
    # driver prints the MDE it is actually buying instead of quoting a comment.
    parser.add_argument("--paired-n", type=int, default=None)
    # Likewise the sign-test table: the binding constraint is SCENES, and a sweep over 19
    # of them should see the row for 19 rather than interpolate between 15 and 20.
    parser.add_argument("--n-scenes", type=int, default=None)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--seconds-per-episode", type=float, default=27.0)
    args = parser.parse_args(None if argv is None else list(argv))

    z_alpha, z_beta = _z(args.alpha, args.power)
    print("")
    print("=== between-cell MDE (UNPAIRED: different scenes, different classes) ===")
    print(
        "  alpha={:.2f} two-sided, power={:.2f}, p={:.2f}  ->  multiplier {:.2f} sigma".format(
            args.alpha, args.power, args.p, z_alpha + z_beta
        )
    )
    print(
        "  2 sigma is the SIGNIFICANCE threshold and detects an effect sitting on it half"
    )
    print("  the time. It is not an MDE. Both columns are printed so the gap is visible.")
    print("")
    print(
        "  {:>10s} {:>8s} {:>10s} {:>12s} {:>12s} {:>10s}".format(
            "n/cell", "total", "wall", "SD(diff)", "2 sigma", "MDE"
        )
    )
    candidates = [args.n_per_cell] if args.n_per_cell else [90, 200, 400, 800]
    for n in candidates:
        sd = sd_between_cells(n, args.p)
        hours = 4 * n * args.seconds_per_episode / 3600.0
        print(
            "  {:>10d} {:>8d} {:>9.1f}h {:>11.1f}pt {:>11.1f}pt {:>9.1f}pt".format(
                n, 4 * n, hours, 100 * sd, 200 * sd, 100 * mde_between_cells(
                    n, args.p, args.alpha, args.power
                )
            )
        )

    print("")
    print(
        "  the effect this project has previously measured: {:.1f} points "
        "(the M3 revisit headline)".format(100 * HISTORICAL_EFFECT)
    )
    needed = episodes_for_mde(HISTORICAL_EFFECT, args.p, args.alpha, args.power)
    print(
        "  episodes/cell to detect it at {:.0f}% power: {} ({} total, {:.1f}h)".format(
            100 * args.power,
            needed,
            4 * needed,
            4 * needed * args.seconds_per_episode / 3600.0,
        )
    )
    if args.target:
        n = episodes_for_mde(args.target, args.p, args.alpha, args.power)
        print(
            "  episodes/cell for a {:.1f}-point MDE: {} ({} total, {:.1f}h)".format(
                100 * args.target, n, 4 * n, 4 * n * args.seconds_per_episode / 3600.0
            )
        )

    print("")
    print("=== the scene-level test, which episodes cannot buy ===")
    print("  {:>10s}  {}".format("scenes", "how many must agree for a two-sided sign test"))
    scene_rows = sorted({10, 15, 20, 30, 40} | ({args.n_scenes} if args.n_scenes else set()))
    for n_scenes in scene_rows:
        threshold = sign_test_threshold(n_scenes, args.alpha)
        verdict = (
            "IMPOSSIBLE - even a clean sweep does not reach alpha"
            if threshold is None
            else "{} of {} ({:.0f}%)".format(threshold, n_scenes, 100 * threshold / n_scenes)
        )
        print("  {:>10d}  {}".format(n_scenes, verdict))
    print("")
    print("  Episodes inside a scene share a room, a source and a renderer, so they are not")
    print("  independent. funnel_diff's scene test already disagreed with an episode-level")
    print("  McNemar once; the rule is to report both. More episodes do not add scenes.")
    print("")

    print("=== the PAIRED formula, for contrast -- it does NOT apply to the matrix ===")
    if args.paired_n:
        # Printed FIRST and labelled as the answer, because a driver that asks for it is
        # a paired sweep and the n=365 line below is then the historical reference rather
        # than the number in front of the reader.
        print(
            "  THIS SWEEP, paired at n={} with the measured {:.3f} flip rate:".format(
                args.paired_n, MEASURED_FLIP_RATE
            )
        )
        print(
            "    SD {:.1f} episodes, MDE {:.2f} points. An effect smaller than that is"
            .format(sd_paired(args.paired_n), 100 * mde_paired(args.paired_n))
        )
        print("    NOT resolvable by this sweep at {:.0f}% power, however it is read.".format(
            100 * args.power
        ))
        print("")
    print(
        "  a byte-identical re-run at n=365 with the measured {:.3f} flip rate:".format(
            MEASURED_FLIP_RATE
        )
    )
    print(
        "    SD {:.1f} episodes, MDE {:.1f} points, against {:.1f} points unpaired at the".format(
            sd_paired(365), 100 * mde_paired(365), 100 * mde_between_cells(365)
        )
    )
    print(
        "    same total rendering cost -- {:.2f}x, because the same episode is in both "
        "arms.".format(mde_between_cells(365) / mde_paired(365))
    )
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
