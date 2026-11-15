"""
Extra calculations used in the discussion (Sections 8.1-8.3).

These calculations do not use the simulator. They illustrate the form of the relationships
discussed in Chapter 8 under simplifying assumptions, stated for each calculation below.
A fixed random seed is used so that the results are reproducible.
"""

import numpy as np

rng = np.random.default_rng(seed=1)

# Simplified dissonance weights per interval class (Table 5.1)
WEIGHTS = np.array([0.0, 1.0, 0.6, 0.2, 0.1, 0.1, 0.8, 0.0, 0.4, 0.1, 0.5, 0.7])

N_AGENTS = 20
WORLD = 500
MIN_NOTE, MAX_NOTE = 36, 96


def pitch_from_x(x):
    """
    Pitch mapping used in the simulator: linear in x and truncated to an integer.

    Positions are not clamped to the world, so boundary effects are ignored.
    """
    return np.floor(MIN_NOTE + x * (MAX_NOTE - MIN_NOTE) / WORLD).astype(int)


# Expected CV_IOI of merged onset sequences (Section 8.2)
# Assumptions: each agent fires strictly periodically; onsets of all agents are merged
# into one sequence of 90 s; results are averaged over repeated draws.
def merged_cv(periods, offsets, duration=90.0, jitter=0.0):
    onsets = []
    for period, offset in zip(periods, offsets):
        times = np.arange(offset * period, duration, period)
        onsets.append(times + rng.normal(0, jitter, len(times)))
    intervals = np.diff(np.sort(np.concatenate(onsets)))
    return intervals.std(ddof=1) / intervals.mean()


def section_8_2_cv():
    print("1. Expected CV_IOI of merged onsets (N = 20, period 2 s)")
    print(f"   Analytical value for random phase offsets, sqrt((N-1)/(N+1)): "
          f"{np.sqrt((N_AGENTS - 1) / (N_AGENTS + 1)):.3f}")
    scenarios = {
        "Identical periods, random phases": lambda: merged_cv(np.full(N_AGENTS, 2.0), rng.random(N_AGENTS)),
        "Periods 2-4 s, random phases": lambda: merged_cv(rng.uniform(2, 4, N_AGENTS), rng.random(N_AGENTS)),
        # Synchronised agents fire within microseconds of each other, modelled with a tiny jitter
        "Four synchronised clusters": lambda: merged_cv(np.full(N_AGENTS, 2.0),
                                                        np.repeat(rng.random(4), N_AGENTS // 4), jitter=1e-5),
        "Full synchrony": lambda: merged_cv(np.full(N_AGENTS, 2.0), np.zeros(N_AGENTS), jitter=1e-5),
        "Evenly spaced phases (splay state)": lambda: merged_cv(np.full(N_AGENTS, 2.0),
                                                                np.arange(N_AGENTS) / N_AGENTS),
    }
    for name, draw in scenarios.items():
        print(f"   {name:<40} CV_IOI = {np.mean([draw() for _ in range(2000)]):.3f}")


# Expected global polarity of k independently moving clusters (Section 8.2)
# Assumptions: k equally sized clusters; agents within a cluster share one direction;
# cluster directions are independent and uniformly distributed in 3D.
def section_8_2_polarity():
    print("\n2. Expected global polarity of k internally aligned clusters")
    for k in [1, 2, 3, 4, 5, 6, 20]:
        directions = rng.normal(size=(200_000, k, 3))
        directions /= np.linalg.norm(directions, axis=2, keepdims=True)
        polarity = (np.linalg.norm(directions.sum(axis=1), axis=1) / k).mean()
        label = "(20 independent agents)" if k == 20 else ""
        print(f"   k = {k:<3} polarity = {polarity:.2f} {label}")



# Expected dissonance as a function of distance (Section 8.3)
# Assumptions: agent positions uniform along x; the direction between two agents is
# uniformly random in 3D, so only the x-component of their distance affects pitch;
# boundary effects are ignored; dissonance uses the pitch-class lookup table.
def section_8_3_dissonance():
    print("\n3. Expected pitch-class dissonance")
    x = rng.uniform(0, WORLD, (2_000_000, 2))
    p = pitch_from_x(x)
    print(f"   Two randomly placed agents: {WEIGHTS[np.abs(p[:, 0] - p[:, 1]) % 12].mean():.2f}")
    print(f"   Mean of the lookup table:   {WEIGHTS.mean():.3f}")
    for distance in [20, 30, 50]:
        cos_theta = rng.uniform(-1, 1, 1_000_000)
        x1 = rng.uniform(0, WORLD, 1_000_000)
        x2 = x1 + distance * cos_theta
        interval = np.abs(pitch_from_x(x1) - pitch_from_x(x2)) % 12
        print(f"   Pair {distance} units apart:      {WEIGHTS[interval].mean():.2f}")


# Expected number of sounding notes as a function of speed (Section 8.1)
# Period of one agent: 1 / (0.0125 |v|) frames x 0.05 s = 4 / |v| s.
# Fraction of time sounding: 0.5 s / (4 / |v| s) = 0.125 |v|.
# Expected sounding notes for 20 agents: 20 x 0.125 |v| = 2.5 |v|.
def section_8_1_speed():
    print("\n4. Expected number of sounding notes, 2.5 x |v|")
    for speed in [0.5, 1.0, 1.5, 2.0]:
        print(f"   |v| = {speed:.1f} units/frame: period {4 / speed:.1f} s, "
              f"expected sounding notes {2.5 * speed:.2f}")


if __name__ == "__main__":
    section_8_2_cv()
    section_8_2_polarity()
    section_8_3_dissonance()
    section_8_1_speed()
