"""
Supplemental calculations (Appendix A), referred to in the discussion (Sections 7.1-7.3).

These calculations do not use the simulator. They provide reference values under the
simplifying assumptions stated for each calculation below, and reproduce the values in
Tables A.2-A.5 and Figure A.1. Values without a closed-form expression are estimated by
Monte Carlo simulation. A fixed random seed is used so that the results are reproducible.
"""

import os
import numpy as np

rng = np.random.default_rng(seed=1)

# Simplified dissonance weights per interval class (Table 4.1)
WEIGHTS = np.array([0.0, 1.0, 0.6, 0.2, 0.1, 0.1, 0.8, 0.0, 0.4, 0.1, 0.5, 0.7])

N_AGENTS = 20
WORLD = 500
MIN_NOTE, MAX_NOTE = 36, 96
DT = 0.0125            # phase increment per frame
FRAME_PERIOD = 0.05    # nominal frame period in seconds
NOTE_DURATION = 0.5    # seconds
SEMITONE_WIDTH = WORLD / (MAX_NOTE - MIN_NOTE)


def pitch_from_x(x):
    """Pitch mapping used in the simulator: linear in x and truncated to an integer."""
    return np.floor(MIN_NOTE + x * (MAX_NOTE - MIN_NOTE) / WORLD).astype(int)


# A.1 Agent speed and the number of sounding notes (Table A.2)
# Firing period T_fire = FRAME_PERIOD / (|v| * DT) = 4 / |v| s.
# Share of time sounding = NOTE_DURATION / T_fire = 0.125 |v|.
# Expected sounding notes M = N * 0.125 |v| = 2.5 |v|, so |v| = M / 2.5.
def appendix_a1_speed():
    print("A.1 Inferred agent speed from the mean number of sounding notes (Table A.2)")
    notes_per_speed = N_AGENTS * NOTE_DURATION * DT / FRAME_PERIOD   # = 2.5
    # Mean number of sounding notes per cohesion weight in scenario 1 (statistics_report.txt)
    observed = {
        "No cohesion, W_coh = 0": 1.3179,
        "Lowest value with cohesion (W_coh = 0.01)": 3.7320,
        "Highest value with cohesion (W_coh = 0.07)": 4.1668,
        "Maximum speed (reference)": notes_per_speed * 2.0,
    }
    for label, m in observed.items():
        speed = m / notes_per_speed
        period = FRAME_PERIOD / (speed * DT)
        print(f"   {label:<44} M = {m:.2f}   |v| = {speed:.2f}   T_fire = {period:.1f} s")


# A.2 Rhythmic regularity of merged onset sequences (Table A.3)
# Assumptions: N perfectly periodic agents with a common period, firing in G equally sized
# synchronised groups with independent, uniformly random phase offsets.
# Closed form: CV_IOI = sqrt(2N / (G + 1) - 1).
def merged_cv(periods, offsets, duration=90.0, jitter=0.0):
    onsets = []
    for period, offset in zip(periods, offsets):
        times = np.arange(offset * period, duration, period)
        onsets.append(times + rng.normal(0, jitter, len(times)))
    intervals = np.diff(np.sort(np.concatenate(onsets)))
    return intervals.std(ddof=1) / intervals.mean()


def appendix_a2_cv():
    print("\nA.2 Expected CV_IOI of merged onsets, N = 20 (Table A.3)")
    for groups in [20, 10, 5, 4, 3, 2, 1]:
        analytical = np.sqrt(2 * N_AGENTS / (groups + 1) - 1)
        # Numerical check: synchronised agents fire within microseconds of each other,
        # modelled with a tiny jitter. Group sizes are as equal as possible.
        labels = np.arange(N_AGENTS) % groups
        # Long sequences (900 s) are used so that edge effects at the start and end are small.
        numerical = np.mean([
            merged_cv(np.full(N_AGENTS, 2.0), rng.random(groups)[labels], duration=900.0,
                      jitter=1e-5 if groups < N_AGENTS else 0.0)
            for _ in range(500)])
        print(f"   G = {groups:<3} analytical = {analytical:.2f}   numerical = {numerical:.2f}")
    # Additional reference cases mentioned in the discussion
    others = {
        "Periods 2-4 s, random phases": lambda: merged_cv(rng.uniform(2, 4, N_AGENTS), rng.random(N_AGENTS)),
        "Evenly spaced phases (splay state)": lambda: merged_cv(np.full(N_AGENTS, 2.0),
                                                                np.arange(N_AGENTS) / N_AGENTS),
    }
    for name, draw in others.items():
        print(f"   {name:<40} CV_IOI = {np.mean([draw() for _ in range(2000)]):.3f}")


# A.3 Global polarity of independently moving clusters (Table A.4)
# Assumptions: G equally sized clusters; agents within a cluster share one direction and
# speed; cluster directions are independent and uniformly distributed in 3D.
# Upper bound: sqrt(E[phi^2]) = 1 / sqrt(G).
def appendix_a3_polarity(repetitions=400_000):
    print("\nA.3 Expected global polarity of G internally aligned clusters (Table A.4)")
    for groups in [1, 2, 3, 4, 5, 10, 20]:
        directions = rng.normal(size=(repetitions, groups, 3))
        directions /= np.linalg.norm(directions, axis=2, keepdims=True)
        polarity = (np.linalg.norm(directions.sum(axis=1), axis=1) / groups).mean()
        print(f"   G = {groups:<3} polarity = {polarity:.2f}   upper bound = {1 / np.sqrt(groups):.2f}")


# A.4 Expected dissonance as a function of distance (Table A.5 and Figure A.1)
# Assumptions: agent A uniform along x; agent B at the given distance in a uniformly random
# 3D direction, so the x-component of the distance is r * cos(beta) with cos(beta) uniform on
# [-1, 1]; pairs falling outside the simulation space are discarded; dissonance uses the
# pitch-class lookup table.
def expected_dissonance(distance, pairs=2_000_000):
    cos_beta = rng.uniform(-1, 1, pairs)
    x1 = rng.uniform(0, WORLD, pairs)
    x2 = x1 + distance * cos_beta
    inside = (x2 >= 0) & (x2 <= WORLD)
    interval = np.abs(pitch_from_x(x1[inside]) - pitch_from_x(x2[inside])) % 12
    return WEIGHTS[interval].mean()


def random_pair_dissonance(pairs=2_000_000):
    x = rng.uniform(0, WORLD, (pairs, 2))
    p = pitch_from_x(x)
    return WEIGHTS[np.abs(p[:, 0] - p[:, 1]) % 12].mean()


def appendix_a4_dissonance(save_figure=True):
    print("\nA.4 Expected dissonance of two agents at a given distance (Table A.5)")
    for distance in [0, 8, 16, 20, 30, 50, 100]:
        print(f"   r = {distance:<4} ({distance / SEMITONE_WIDTH:4.1f} semitone widths)   "
              f"dissonance = {expected_dissonance(distance):.2f}")
    baseline = random_pair_dissonance()
    print(f"   Two randomly placed agents:          dissonance = {baseline:.2f}")
    print(f"   Mean of the lookup table:            {WEIGHTS.mean():.3f}")

    if save_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        distances = np.arange(0, 151)
        values = [expected_dissonance(r, pairs=200_000) for r in distances]
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        ax.plot(distances, values, color="#1f4e9a", lw=1.5, label="Expected dissonance of a pair")
        ax.axhline(baseline, color="0.4", lw=0.9, ls="--", label="Two randomly placed agents")
        for x, label in [(20, "separation distance"), (50, "vision radius")]:
            ax.axvline(x, color="0.6", lw=0.6, ls=":")
            ax.text(x + 1.5, 0.04, label, fontsize=8, rotation=90, va="bottom", color="0.35")
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 0.75)
        ax.set_xlabel("Distance between the two agents (units)")
        ax.set_ylabel("Expected dissonance")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "appendix_dissonance_distance.pdf"))
        print(f"   Figure A.1 saved to {out_dir}")


if __name__ == "__main__":
    appendix_a1_speed()
    appendix_a2_cv()
    appendix_a3_polarity()
    appendix_a4_dissonance()
