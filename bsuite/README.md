# Core RL Behaviour Suite

## Introduction

bsuite is a collection of carefully-designed experiments that investigate core
capabilities of a reinforcement learning (RL) agent with two main objectives.

- First, to collect clear, informative and scalable problems that capture key
  issues in the design of efficient and general learning algorithms.
- Second, to study agent behavior through their performance on these shared
  benchmarks.

This library automates evaluation and analysis of any agent on these benchmarks.
It serves to facilitate reproducible, and accessible, research on the core
issues in RL, and ultimately the design of superior learning algorithms.

Going forward, we hope to incorporate more excellent experiments from the
research community, and commit to a periodic review of the experiments from a
committee of prominent researchers.

## Technical overview

bsuite is a collection of _experiments_, defined in the `experiments`
subdirectory. Each subdirectory corresponds to one experiment and contains:

- A file defining RL environment, which may be configurable to provide different
  levels of difficulty or different random seeds (for example).
- A sequence of keyword arguments for this environment, defined in the
  `SETTINGS` variable found in the experiment's `sweep.py` file.
- A file `analysis.py` defining plots used in the provided Jupyter notebook.

bsuite works by logging results from "within" each environment. This means any
experiment will automatically output data in the correct format for analysis
using the notebook. Logging results from within the environment means that
bsuite does not impose any constraints on the structure of agents or algorithms.

The default method of logging writes to an SQLite database. Users just need to
specify a file path when loading the environment. The notebook generates
detailed analysis from this data.

## Installation

We have tested bsuite on Python 3.5. We do not attempt to maintain a working
version for Python 2.7.

To install bsuite, run the command

```
pip install git+git://github.com/deepmind/bsuite.git
```

or clone the repository and run

```
pip install /path/to/bsuite/
```

To install the package while being able to edit the code (see baselines below),
run

```
pip install -e /path/to/bsuite/
```

## Baselines

We also include implementations of several common agents in the `baselines`
subdirectory, along with a minimal run-loop.

Our baselines additionally depend on [TensorFlow](http://tensorflow.org) and
[Sonnet](https://github.com/deepmind/sonnet). These dependencies are not
installed by default, since bsuite is independent of any machine learning
library.
