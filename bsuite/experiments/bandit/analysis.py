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
"""Analysis for bandit."""

from typing import Sequence

from bsuite.experiments.bandit import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
BASE_REGRET = 0.5
TAGS = sweep.TAGS


def score(df: pd.DataFrame) -> float:
  """Output a single score for bandit experiment."""
  return plotting.ave_regret_score(
      df, baseline_regret=BASE_REGRET, episode=sweep.NUM_EPISODES)


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[str] = None) -> gg.ggplot:
  """Plots the average regret through time."""
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=sweep.NUM_EPISODES)
  return bandit_learning_format(p)


def bandit_learning_format(plot: gg.ggplot) -> gg.ggplot:
  """Add nice bandit formatting to ggplot."""
  plot += gg.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1).tolist())
  plot += gg.theme(panel_grid_major_y=gg.element_line(size=2.5),
                   panel_grid_minor_y=gg.element_line(size=0))
  plot += gg.geom_hline(
      gg.aes(yintercept=BASE_REGRET), linetype='dashed', alpha=0.4, size=1.75)
  plot += gg.coord_cartesian(ylim=(0, 1))
  return plot


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Sequence[str] = None,
               colour_var: str = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = 1.0 - (df.total_regret.diff() / df.episode.diff())
  p = plotting.plot_individual_returns(
      df_in=df,
      max_episode=NUM_EPISODES,
      return_column='average_return',
      colour_var=colour_var,
      yintercept=1.,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')
