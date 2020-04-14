# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Analysis for umbrella_distract experiment."""

from typing import Sequence

from bsuite.experiments.umbrella_distract import sweep
from bsuite.experiments.umbrella_length import analysis as umbrella_length_analysis
from bsuite.utils import plotting
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = sweep.TAGS


def score(df: pd.DataFrame) -> float:
  return umbrella_length_analysis.score_by_group(df, 'n_distractor')


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[str] = None) -> gg.ggplot:
  """Plots the average regret through time."""
  return plotting.plot_regret_group_nosmooth(
      df_in=df,
      group_col='n_distractor',
      sweep_vars=sweep_vars,
      max_episode=sweep.NUM_EPISODES,
  )


def plot_scale(df: pd.DataFrame,
               sweep_vars: Sequence[str] = None) -> gg.ggplot:
  """Plots the average return at end of learning investigating scaling."""
  return plotting.plot_regret_ave_scaling(
      df_in=df,
      group_col='n_distractor',
      episode=sweep.NUM_EPISODES,
      regret_thresh=0.5,
      sweep_vars=sweep_vars,
  )


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Sequence[str] = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  return umbrella_length_analysis.plot_seeds(df_in, sweep_vars, 'n_distractor')
