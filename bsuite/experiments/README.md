# `bsuite` Experiments

This folder contains all of the experiments that constitute `bsuite`. Each
experiment folder contains three files:

1.  A mechanism for loading a RL environment that adheres to the [dm_env](https://github.com/deepmind/dm_env)
    interface (see `environments/` for the precise environment definitions).
1.  `sweep.py`, which contains a list of different configurations of this
    environment over which the agent is tested.
1.  `analysis.py`, which specifies how to 'score' the experiment, and
    provides utilities for generating relevant plots.

Detailed descriptions of each experiment can be found in the Jupyter notebook in
`analysis/results.ipynb`.
