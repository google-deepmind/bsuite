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
"""Analysis for discounting_chain."""

from typing import Optional, Sequence

from bsuite.experiments.discounting_chain import sweep
from bsuite.utils import plotting

import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
BASE_REGRET = 0.08
TAGS = sweep.TAGS
_HORIZONS = np.array([1, 3, 10, 30, 100])


def score(df: pd.DataFrame) -> float:
  """Output a single score for discounting_chain."""
  n_eps = np.minimum(df.episode.max(), sweep.NUM_EPISODES)
  ave_return = df.loc[df.episode == n_eps, 'total_return'].mean() / n_eps
  raw_score = 1. - 10. * (1.1 - ave_return)
  return np.clip(raw_score, 0, 1)


def _mapping_seed_compatibility(df: pd.DataFrame) -> pd.DataFrame:
  """Utility function to maintain compatibility with old bsuite runs."""
  # Discounting chain kwarg "seed" was renamed to "mapping_seed"
  if 'mapping_seed' in df.columns:
    nan_seeds = df.mapping_seed.isna()
    if np.any(nan_seeds):
      df.loc[nan_seeds, 'mapping_seed'] = df.loc[nan_seeds, 'seed']
      print('WARNING: seed renamed to "mapping_seed" for compatibility.')
  else:
    if 'seed' in df.columns:
      print('WARNING: seed renamed to "mapping_seed" for compatibility.')
      df['mapping_seed'] = df.seed
    else:
      print('ERROR: outdated bsuite run, please relaunch.')
  return df


def dc_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess discounting chain data for use with regret metrics."""
  df = df_in.copy()
  df = _mapping_seed_compatibility(df)
  df['optimal_horizon'] = _HORIZONS[
      (df.mapping_seed % len(_HORIZONS)).astype(int)]
  df['total_regret'] = 1.1 * df.episode - df.total_return
  df['optimal_horizon'] = df.optimal_horizon.astype('category')
  return df


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plots the average regret through time by optimal_horizon."""
  df = dc_preprocess(df_in=df)
  p = plotting.plot_regret_learning(
      df_in=df,
      group_col='optimal_horizon',
      sweep_vars=sweep_vars,
      max_episode=sweep.NUM_EPISODES
  )
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  p += gg.coord_cartesian(ylim=(0, 0.1))
  return p


def plot_average(df: pd.DataFrame,
                 sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plots the average regret at 1k episodes by optimal_horizon."""
  df = dc_preprocess(df_in=df)
  p = plotting.plot_regret_average(
      df_in=df,
      group_col='optimal_horizon',
      episode=sweep.NUM_EPISODES,
      sweep_vars=sweep_vars
  )
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = dc_preprocess(df_in)
  df['average_return'] = 1.1 - (df.total_regret.diff() / df.episode.diff())
  p = plotting.plot_individual_returns(
      df_in=df,
      max_episode=NUM_EPISODES,
      return_column='average_return',
      colour_var='optimal_horizon',
      yintercept=1.1,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')
