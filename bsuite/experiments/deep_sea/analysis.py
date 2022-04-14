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
"""Analysis for deep_sea experiment."""

from typing import Optional, Sequence

from bsuite.experiments.deep_sea import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = sweep.TAGS


def _check_data(df: pd.DataFrame) -> None:
  """Check that the data has the correct information logged."""
  assert 'total_bad_episodes' in df.columns
  assert 'episode' in df.columns
  assert 'size' in df.columns


def find_solution(df_in: pd.DataFrame,
                  sweep_vars: Optional[Sequence[str]] = None,
                  merge: bool = True,
                  thresh: float = 0.8,
                  num_episodes: int = NUM_EPISODES) -> pd.DataFrame:
  """Find first episode that gets below thresh regret by sweep_vars."""
  # Check data has the necessary columns for deep sea
  df = df_in.copy()
  _check_data(df)
  df = df[df.episode <= num_episodes]

  # Parse the variables that you are aggregating over
  if sweep_vars is None:
    sweep_vars = ['size']
  elif 'size' not in sweep_vars:
    sweep_vars = list(sweep_vars) + ['size']

  # Find the earliest episode that gets at least below thresh regret
  df['avg_bad_episodes'] = df.total_bad_episodes / df.episode
  plt_df = df[df.avg_bad_episodes < thresh].groupby(sweep_vars)['episode']
  plt_df = plt_df.min().reset_index()

  solved = plt_df.set_index(sweep_vars).episode
  unsolved_ids = set(df.set_index(sweep_vars).index) - set(solved.index)
  unsolved = df.groupby(sweep_vars)['episode'].max()[list(unsolved_ids)]

  plt_df = solved.append(unsolved).to_frame()
  plt_df.rename(columns={0: 'episode'}, inplace=True)
  plt_df.loc[solved.index, 'solved'] = True
  plt_df.loc[unsolved.index, 'solved'] = False
  plt_df.rename(columns={0: 'episode'}, inplace=True)
  plt_df.reset_index(inplace=True)

  # Add a column to see if the experiment has finished 10k episodes
  finish_df = (
      df.groupby(sweep_vars)['episode'].max() >= num_episodes).reset_index()
  finish_df.rename(columns={'episode': 'finished'}, inplace=True)
  plt_df = plt_df.merge(finish_df, on=sweep_vars)
  plt_df.loc[plt_df.solved, 'finished'] = True  # If solved -> finished

  # Optionally merge back with all the df columns.
  if merge:
    join_vars = sweep_vars + ['episode']
    plt_df = plt_df.merge(df, on=join_vars)

  return plt_df


def score(df: pd.DataFrame,
          forgiveness: float = 100.) -> float:
  """Outputs a single score for deep sea selection."""
  plt_df = find_solution(df)
  beat_dither = (plt_df.solved
                 & (plt_df.episode < 2 ** plt_df['size'] + forgiveness))
  return np.mean(beat_dither)


def _make_baseline(plt_df: pd.DataFrame,
                   sweep_vars: Optional[Sequence[str]] = None) -> pd.DataFrame:
  """Generate baseline 2^N data for each combination of sweep_vars."""
  x = np.arange(5, 20)
  baseline = pd.DataFrame(dict(size=x, episode=2**x))
  if sweep_vars:
    params = plt_df.groupby(sweep_vars).size().reset_index().drop(0, axis=1)
    data = []
    for _, row in params.iterrows():
      tmp = baseline.copy()
      for col, val in row.iteritems():
        tmp[col] = val
      data.append(tmp)
    return pd.concat(data, sort=True)
  else:
    return baseline


def _base_scaling(plt_df: pd.DataFrame,
                  sweep_vars: Optional[Sequence[str]] = None,
                  with_baseline: bool = True) -> gg.ggplot:
  """Base underlying piece of the scaling plots for deep sea."""
  p = (gg.ggplot(plt_df)
       + gg.aes(x='size', y='episode')
      )
  if np.all(plt_df.finished):
    p += gg.geom_point(gg.aes(colour='solved'), size=3, alpha=0.75)
  else:
    p += gg.geom_point(gg.aes(shape='finished', colour='solved'),
                       size=3, alpha=0.75)
    p += gg.scale_shape_manual(values=['x', 'o'])

  if np.all(plt_df.solved):
    p += gg.scale_colour_manual(values=['#313695'])  # blue
  else:
    p += gg.scale_colour_manual(values=['#d73027', '#313695'])  # [red, blue]

  if with_baseline:
    baseline_df = _make_baseline(plt_df, sweep_vars)
    p += gg.geom_line(data=baseline_df, colour='black',
                      linetype='dashed', alpha=0.4, size=1.5)
  return p


def plot_scaling(plt_df: pd.DataFrame,
                 sweep_vars: Optional[Sequence[str]] = None,
                 with_baseline: bool = True,
                 num_episodes: int = NUM_EPISODES) -> gg.ggplot:
  """Plot scaling of learning time against exponential baseline."""
  p = _base_scaling(plt_df, sweep_vars, with_baseline)
  p += gg.xlab('deep sea problem size')
  p += gg.ylab('#episodes until < 90% bad episodes')
  if with_baseline:
    max_steps = np.minimum(num_episodes, plt_df.episode.max())
    p += gg.coord_cartesian(ylim=(0, max_steps))
  return plotting.facet_sweep_plot(p, sweep_vars)


def plot_scaling_log(plt_df: pd.DataFrame,
                     sweep_vars: Optional[Sequence[str]] = None,
                     with_baseline=True) -> gg.ggplot:
  """Plot scaling of learning time against exponential baseline."""
  p = _base_scaling(plt_df, sweep_vars, with_baseline)
  p += gg.scale_x_log10(breaks=[5, 10, 20, 50])
  p += gg.scale_y_log10(breaks=[100, 300, 1000, 3000, 10000, 30000])
  p += gg.xlab('deep sea problem size (log scale)')
  p += gg.ylab('#episodes until < 90% bad episodes (log scale)')
  return plotting.facet_sweep_plot(p, sweep_vars)


def plot_regret(df_in: pd.DataFrame,
                sweep_vars: Optional[Sequence[str]] = None,
                num_episodes: int = NUM_EPISODES) -> gg.ggplot:
  """Plot average regret of deep_sea through time by size."""
  df = df_in.copy()
  df = df[df['size'].isin([10, 20, 30, 40, 50])]
  df['avg_bad'] = df.total_bad_episodes / df.episode
  df['size'] = df['size'].astype('category')
  p = (gg.ggplot(df[df.episode <= num_episodes])
       + gg.aes('episode', 'avg_bad', group='size', colour='size')
       + gg.geom_line(size=2, alpha=0.75)
       + gg.geom_hline(
           gg.aes(yintercept=0.99), linetype='dashed', alpha=0.4, size=1.75)
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
       + gg.ylab('average bad episodes')
       + gg.scale_colour_manual(values=plotting.FIVE_COLOURS)
      )
  return plotting.facet_sweep_plot(p, sweep_vars)


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Optional[Sequence[str]] = None,
               yintercept: float = 0.99,
               num_episodes: int = NUM_EPISODES) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = df.denoised_return.diff() / df.episode.diff()
  p = plotting.plot_individual_returns(
      df_in=df[df.episode > 0.01 * num_episodes],  # First episodes very noisy
      max_episode=num_episodes,
      return_column='average_return',
      colour_var='size',
      yintercept=yintercept,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')
