# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Process to add new experiments in BSuite:
1.  If you're creating a completely new environment, create a directory in `bsuite/environments` with: `<env>.py`. `<env>.py` should define a new env_class which should be a subclass of `bsuite.environments.base.Environment` (and it should return appropriate info in `bsuite_info()`).

1.  Create directory in `bsuite/experiments` with: `<exp>.py`, `sweep.py`, `analysis.py`, `__init__.py`, `<exp>_test.py`.
    *  `<exp>.py`: Needs to import the environment used for the experiment that is defined in `bsuite/environments/`
    *  `<env>.py` and define a load variable in the file that is equal to `<env_class`, i.e., `load = <env>.<env_class>`
    *  `sweep.py`: Needs to have the parameters that vary for the experiment. e.g., `seed` and `noise_scale` for `cartpole_noise`. Each set of parameters is stored as a dict in a tuple named `SETTINGS`. This file also defines `NUM_EPISODES` and `TAGS` (such as `credit_assignment`, `basic`, `exploration`, etc.). In `TAGS`, the 1st tag should be one of the basic "types" from `summary_analysis.py`: `['basic', 'noise', 'scale', 'exploration', 'credit_assignment', 'memory', 'mdp_playground']`. NOTE: Remember to add a comma after the tag in `TAGS` if there is only 1 tag, because the comma is what makes it a tuple in Python.
    *  `analysis.py`: Needs to define `score()`, `plot_learning()`, `plot_seeds()` (and possibly other functions like `plot_average`) that will be used by `bsuite/analysis/results.ipynb` to analyse and plot recorded data.

1.  `bsuite/bsuite.py`, `bsuite/sweep.py`, `bsuite/experiments/summary_analysis.py` and `bsuite/analysis/results.ipynb` need to be modified for each new experiment added. We need to add code lines specific to the new experiment, e.g., `from bsuite.experiments.<exp> import ...`.
