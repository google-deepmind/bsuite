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
"""Analysis functions for mountain_car experiment."""

from typing import Sequence

from bsuite.experiments.mountain_car import sweep
from bsuite.utils import plotting
import pandas as pd
import plotnine as gg

_SOLVED_STEPS = 25
NUM_EPISODES = sweep.NUM_EPISODES
TAGS = sweep.TAGS
BASE_REGRET = 415  # Regret of the random policy empirically


def score(df: pd.DataFrame) -> float:
  """Output a single score for mountain car."""
  cp_df = mountain_car_preprocess(df_in=df)
  return plotting.ave_regret_score(
      cp_df, baseline_regret=BASE_REGRET, episode=sweep.NUM_EPISODES)


def mountain_car_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess mountain_car data for use with regret metrics."""
  df = df_in.copy()
  ideal_total_return = _SOLVED_STEPS * -1 * df.episode
  total_return = df.raw_return  # Sum of all rewards so far.
  df['total_regret'] = ideal_total_return - total_return
  return df


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[str] = None) -> gg.ggplot:
  """Simple learning curves for mountain_car."""
  df = mountain_car_preprocess(df)
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=sweep.NUM_EPISODES)
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Sequence[str] = None,
               colour_var: str = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = df.raw_return.diff() / df.episode.diff()
  p = plotting.plot_individual_returns(
      df_in=df,
      max_episode=NUM_EPISODES,
      return_column='average_return',
      colour_var=colour_var,
      yintercept=-_SOLVED_STEPS,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')

