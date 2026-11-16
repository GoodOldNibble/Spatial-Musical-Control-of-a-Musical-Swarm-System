# Spatial-Musical Control of a Musical Swarm System

Code and data for the master's thesis *Spatial-Musical Control of a Musical Swarm System: On how spatial metrics correlate to musical structures*
(Master's thesis in Informatics: Robotics and Artificial Intelligence, University of Oslo, 2026).

The thesis examines how spatial properties of a three-dimensional Boids swarm relate to the
musical structures produced when each agent is mapped to a MIDI voice. Release `v1.2.0`
contains the exact code and data that the results in the thesis are based on.

## Contents

| File / folder | Description |
|---|---|
| `MusicalSwarmSim5.py` | The simulator: Boids swarm, agent-centric MIDI sonification and logging of spatial and musical descriptors (thesis Chapter 4). |
| `RunAggregator.py` | Reads the raw output of all runs, discards the warm-up period and reduces each run to one row of mean values (thesis Section 5.4). |
| `DataAnalysis.py` | Statistical tests and figures reported in thesis Chapter 6, and the report `statistics_report.txt` with every statistical result reported in Chapters 6 and 7. |
| `ExtraCalculations.py` | The supplemental calculations in thesis Appendix A (Tables A.2–A.5 and Figure A.1). Does not use the simulator. |
| `statistics_report.txt` | Output of `DataAnalysis.py`: all statistical results reported in the thesis. |
| `Data runs/` | Raw output of all 200 runs, two files per run (see below). |
| `Data analysis/` | Aggregated per-run datasets produced by `RunAggregator.py`. |
| `LICENSE` | MIT licence. |

## Requirements

The experiments were run on Windows 11 with Python 3.9.19. Install the dependencies with

```
pip install -r requirements.txt
```

The simulator sends MIDI to a virtual output port named `loopMIDI Port 1`
([loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) on Windows). Change the
port name in `MusicMapper.__init__` if another port is used. **Note events, and thereby all
musical descriptors, are only generated when a MIDI output port is available.** Connecting a
digital audio workstation to the port is only needed for listening to the output.

## Reproducing the results

1. **Simulation.** Set `BATCH_RUN = True` in `MusicalSwarmSim5.py`. The batch loop runs one
   parameter sweep at a time (five repetitions of ten weights, 2,100 frames per run). To run
   another sweep, replace the increment list, file names and weight variable in the loop, as
   described in the comment above the sweep definitions. Output is written to `runs/`; move
   the files to `Data runs/` before aggregating. With `BATCH_RUN = False` the simulator runs
   interactively until the window is closed.
2. **Aggregation.** Run `RunAggregator.py`. It writes one row per run to
   `Data analysis/master_results_w_notes.csv`.
3. **Analysis.** Run `DataAnalysis.py`. It prints the planned tests, saves the figures to
   `plots/` and writes `statistics_report.txt`.
4. **Supplemental calculations.** Run `ExtraCalculations.py` to reproduce Appendix A.

Because the swarm starts from random initial conditions and the simulation runs in real time,
new simulation runs will not reproduce the raw data exactly. Steps 2–4 reproduce the reported
statistics exactly from the data in this repository.

## Data

Each run in `Data runs/` is named `Batch<b>_Run<r>_<RQ>_<Parameter>_<weight>_<type>.csv`:

- `*_metrics.csv`: one row per frame with `swarm_volume` (convex hull volume),
  `global_polarity`, `mean_nnd` (average nearest neighbour distance), `num_active_notes`,
  `system_dissonance`, `ambitus` (semitones), the three steering weights and a `timestamp`
  in seconds.
- `*_rhythm.csv`: one row per note onset with `ioi` (inter-onset interval in seconds,
  merged across all agents), `polarity` at the onset and `timestamp`.

The aggregated datasets in `Data analysis/` contain one row per run with `batch`, `run_id`,
`rq`, `parameter`, `weight`, the mean of each descriptor over the analysed frames,
`cv_ioi` (coefficient of variation of the inter-onset intervals) and `active_notes`
(mean number of sounding notes).

## Licence

MIT, see `LICENSE`.
