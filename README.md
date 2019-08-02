# Core RL Behaviour Suite (bsuite)

## Introduction

`bsuite` is a collection of carefully-designed experiments that investigate core
capabilities of a reinforcement learning (RL) agent with two main objectives.

1.  To collect clear, informative and scalable problems that capture key issues
    in the design of efficient and general learning algorithms.
2.  To study agent behavior through their performance on these shared
    benchmarks.

This library automates evaluation and analysis of any agent on these benchmarks.
It serves to facilitate reproducible, and accessible, research on the core
issues in RL, and ultimately the design of superior learning algorithms.

Going forward, we hope to incorporate more excellent experiments from the
research community, and commit to a periodic review of the experiments from a
committee of prominent researchers.

## Technical overview

`bsuite` is a collection of _experiments_, defined in the `experiments`
subdirectory. Each subdirectory corresponds to one experiment and contains:

-   A file defining an RL environment, which may be configurable to provide
    different levels of difficulty or different random seeds (for example).
-   A sequence of keyword arguments for this environment, defined in the
    `SETTINGS` variable found in the experiment's `sweep.py` file.
-   A file `analysis.py` defining plots used in the provided Jupyter notebook.

bsuite works by logging results from "within" each environment. This means any
experiment will automatically output data in the correct format for analysis
using the notebook. Logging results from within the environment means that
bsuite does not impose any constraints on the structure of agents or algorithms.

The default method of logging writes to an SQLite database. Users just need to
specify a file path when loading the environment. The notebook generates
detailed analysis from this data.

## Getting started

### Installation

We have tested `bsuite`on Python 3.5. We do not attempt to maintain a working
version for Python 2.7.

To install `bsuite`, run the command

```
pip install git+git://github.com/deepmind/bsuite.git
```

or clone the repository and run

```
pip install bsuite/
```

To install the package while being able to edit the code (see baselines below),
run

```
pip install -e bsuite/
```

To also install dependencies for the `baselines/` examples (excluding Gym and
Dopamine examples), install with:

```
pip install -e bsuite[baselines]
```

### Loading an environment

Environments are specified by a string (e.g. 'catch', 'deep_sea', etc.) followed
by an experiment number, which specifies a particular configuration of the
environment. Together this forms the `bsuite_id` for an experiment.

```python
import bsuite

env = bsuite.load_from_id('catch/0')
```

### Interacting with an environment

Our environments implement the Python interface defined in
[`dm_env`](https://github.com/deepmind/dm_env).

More specifically, all our environments accept a discrete, zero-based integer
action (or equivalently, a scalar numpy array with shape `()`).

To determine the number of actions for a specific environment, use

```python
num_actions = env.action_spec().num_values
```

Each environment returns observations in the form of a numpy array.

We also expose a `bsuite_num_episodes` property for each environment in bsuite.
This allows users to run exactly the number of episodes required for bsuite's
analysis, which may vary between environments used in different experiments.

Example run loop for an agent with a `step()` method.

```python
for _ in range(env.bsuite_num_episodes):
  timestep = env.reset()
  while not timestep.last():
    action = agent.step(timestep)
    timestep = env.step(action)
  agent.step(timestep)
```

## Environments

These environments all have small observation sizes such that you should expect
reasonable performance running on a CPU.

### Environment descriptions

<!-- 'Ground truth' of environment descriptions here? -->

### Rolling your own environment into `bsuite`

<!-- Describe what API your environment needs to fulfil -->

## Baseline agents

We include implementations of several common agents in the `baselines`
subdirectory, along with a minimal run-loop.

See the [installation][#installation] section for how to include the required
dependencies (mainly [TensorFlow](http://tensorflow.org) and [Sonnet](https://github.com/deepmind/sonnet)) at install time. These dependencies are
not installed by default, since `bsuite` does not require users to use any
specific machine learning library.

## Running the entire suite of experiments

See `baselines/scripts/run_sweep.py` for an example of running a baseline agent
against the entire suite of experiments. On a 12 core machine, this will take
around an hour to complete for most agents.

### Logging

We include environment wrappers for handling the logging:

*   A CSV logger in `logging/csv_logging.py`
*   A SQLite logger in `logging/sqlite_logging.py`
*   A terminal logger in `logging/terminal_logging.py`

The most convenient way to use this is via `bsuite.load_and_record_to_csv` or
`bsuite.load_and_record_to_sqlite`.

### Analysis

`bsuite` comes with a ready-made analysis colab included in
`analysis/results.ipynb`. This colab loads and processes logged data, and
produces the scores and plots for each experiment.

### Bsuite Report

You can generate a short PDF report summarizing how different algorithms compare
along the dimensions considered by Bsuite by simply running `pdflatex
bsuite/reports/bsuite_report.tex`.

Examples of bsuite reports can be found in `bsuite/reports/examples/`.
