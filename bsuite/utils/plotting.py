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
"""Common plotting and analysis code.

This code is based around plotnine = python implementation of ggplot.
Typically, these plots will be imported and used within experiment analysis.
"""

from typing import Callable, Optional, Sequence

from bsuite.utils import smoothers
import matplotlib.style as style
import numpy as np
import pandas as pd
import plotnine as gg

# Updates the theme to preferred default settings
gg.theme_set(gg.theme_bw(base_size=18, base_family='serif'))
gg.theme_update(figure_size=(12, 8), panel_spacing_x=0.5, panel_spacing_y=0.5)
style.use('seaborn-poster')
style.use('ggplot')

FIVE_COLOURS = [
    '#313695',  # DARK BLUE
    '#74add1',  # LIGHT BLUE
    '#4daf4a',  # GREEN
    '#f46d43',  # ORANGE
    '#d73027',  # RED
] * 10  # Hack to allow internal code to use functions without error

CATEGORICAL_COLOURS = ([
    '#313695',  # DARK BLUE
    '#74add1',  # LIGHT BLUE
    '#4daf4a',  # GREEN
    '#f46d43',  # ORANGE
    '#d73027',  # RED
    '#984ea3',  # PURPLE
    '#f781bf',  # PINK
    '#ffc832',  # YELLOW
    '#000000',  # BLACK
]) * 100  # For very large sweeps the colours will just have to repeat.


def ave_regret_score(df: pd.DataFrame,
                     baseline_regret: float,
                     episode: int,
                     regret_column: str = 'total_regret') -> float:
  """Score performance by average regret, normalized to [0,1] by baseline."""
  n_eps = np.minimum(df.episode.max(), episode)
  mean_regret = df.loc[df.episode == n_eps, regret_column].mean() / n_eps
  unclipped_score = (baseline_regret - mean_regret) / baseline_regret
  return np.clip(unclipped_score, 0, 1)


def score_by_scaling(df: pd.DataFrame,
                     score_fn: Callable[[pd.DataFrame], float],
                     scaling_var: str) -> float:
  """Apply scoring function based on mean and std."""
  scores = []
  for _, sub_df in df.groupby(scaling_var):
    scores.append(score_fn(sub_df))
  mean_score = np.clip(np.mean(scores), 0, 1)
  lcb_score = np.clip(np.mean(scores) - np.std(scores), 0, 1)
  return 0.5 * (mean_score + lcb_score)


def facet_sweep_plot(base_plot: gg.ggplot,
                     sweep_vars: Optional[Sequence[str]] = None,
                     tall_plot: bool = False) -> gg.ggplot:
  """Add a facet_wrap to the plot based on sweep_vars."""
  df = base_plot.data.copy()

  if sweep_vars:
    # Work out what size the plot should be based on the hypers + add facet.
    n_hypers = df[sweep_vars].drop_duplicates().shape[0]
    base_plot += gg.facet_wrap(sweep_vars, labeller='label_both')
  else:
    n_hypers = 1

  if n_hypers == 1:
    fig_size = (7, 5)
  elif n_hypers == 2:
    fig_size = (13, 5)
  elif n_hypers == 4:
    fig_size = (13, 8)
  elif n_hypers <= 12:
    fig_size = (15, 4 * np.divide(n_hypers, 3) + 1)
  else:
    print('WARNING - comparing {} agents at once is more than recommended.'
          .format(n_hypers))
    fig_size = (15, 12)

  if tall_plot:
    fig_size = (fig_size[0], fig_size[1] * 1.25)

  theme_settings = gg.theme_bw(base_size=18, base_family='serif')
  theme_settings += gg.theme(
      figure_size=fig_size, panel_spacing_x=0.5, panel_spacing_y=0.5,)

  return base_plot + theme_settings


def plot_regret_learning(df_in: pd.DataFrame,
                         group_col: Optional[str] = None,
                         sweep_vars: Optional[Sequence[str]] = None,
                         regret_col: str = 'total_regret',
                         max_episode: Optional[int] = None) -> gg.ggplot:
  """Plots the average regret through time, grouped by group_var."""
  df = df_in.copy()
  df['average_regret'] = df[regret_col] / df.episode
  df = df[df.episode <= (max_episode or np.inf)]
  if group_col is None:
    p = _plot_regret_single(df)
  else:
    p = _plot_regret_group(df, group_col)
  p += gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
  p += gg.ylab('average regret per timestep')
  p += gg.coord_cartesian(xlim=(0, max_episode))
  return facet_sweep_plot(p, sweep_vars, tall_plot=True)


def _plot_regret_single(df: pd.DataFrame) -> gg.ggplot:
  """Plots the average regret through time for single variable."""
  p = (gg.ggplot(df)
       + gg.aes(x='episode', y='average_regret')
       + gg.geom_smooth(method=smoothers.mean, span=0.1, size=1.75, alpha=0.1,
                        colour='#313695', fill='#313695'))
  return p


def _plot_regret_group(df: pd.DataFrame, group_col: str) -> gg.ggplot:
  """Plots the average regret through time when grouped."""
  group_name = group_col.replace('_', ' ')
  df[group_name] = df[group_col].astype('category')
  p = (gg.ggplot(df)
       + gg.aes(x='episode', y='average_regret',
                group=group_name, colour=group_name, fill=group_name)
       + gg.geom_smooth(method=smoothers.mean, span=0.1, size=1.75, alpha=0.1)
       + gg.scale_colour_manual(values=FIVE_COLOURS)
       + gg.scale_fill_manual(values=FIVE_COLOURS))
  return p


def plot_regret_group_nosmooth(df_in: pd.DataFrame,
                               group_col: str,
                               sweep_vars: Optional[Sequence[str]] = None,
                               regret_col: str = 'total_regret',
                               max_episode: Optional[int] = None) -> gg.ggplot:
  """Plots the average regret through time without smoothing."""
  df = df_in.copy()
  df['average_regret'] = df[regret_col] / df.episode
  df = df[df.episode <= max_episode]
  group_name = group_col.replace('_', ' ')
  df[group_name] = df[group_col]
  p = (gg.ggplot(df)
       + gg.aes(x='episode', y='average_regret',
                group=group_name, colour=group_name)
       + gg.geom_line(size=2, alpha=0.75)
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
      )
  p += gg.coord_cartesian(xlim=(0, max_episode))
  return facet_sweep_plot(p, sweep_vars, tall_plot=True)


def _preprocess_ave_regret(df_in: pd.DataFrame,
                           group_col: str,
                           episode: int,
                           sweep_vars: Optional[Sequence[str]] = None,
                           regret_col: str = 'total_regret') -> gg.ggplot:
  """Preprocess the data at episode for average regret calculations."""
  df = df_in.copy()
  group_vars = (sweep_vars or []) + [group_col]
  plt_df = (df[df.episode == episode]
            .groupby(group_vars)[regret_col].mean().reset_index())
  if len(plt_df) == 0:  # pylint:disable=g-explicit-length-test
    raise ValueError('Your experiment has not yet run the necessary {} episodes'
                     .format(episode))
  group_name = group_col.replace('_', ' ')
  plt_df[group_name] = plt_df[group_col].astype('category')
  plt_df['average_regret'] = plt_df[regret_col] / episode
  return plt_df


def plot_regret_average(df_in: pd.DataFrame,
                        group_col: str,
                        episode: int,
                        sweep_vars: Optional[Sequence[str]] = None,
                        regret_col: str = 'total_regret') -> gg.ggplot:
  """Bar plot the average regret at end of learning."""
  df = _preprocess_ave_regret(df_in, group_col, episode, sweep_vars, regret_col)
  group_name = group_col.replace('_', ' ')
  p = (gg.ggplot(df)
       + gg.aes(x=group_name, y='average_regret', fill=group_name)
       + gg.geom_bar(stat='identity')
       + gg.scale_fill_manual(values=FIVE_COLOURS)
       + gg.ylab('average regret after {} episodes'.format(episode))
      )
  return facet_sweep_plot(p, sweep_vars)


def plot_regret_ave_scaling(df_in: pd.DataFrame,
                            group_col: str,
                            episode: int,
                            regret_thresh: float,
                            sweep_vars: Optional[Sequence[str]] = None,
                            regret_col: str = 'total_regret') -> gg.ggplot:
  """Point plot of average regret investigating scaling to threshold."""
  df = _preprocess_ave_regret(df_in, group_col, episode, sweep_vars, regret_col)
  group_name = group_col.replace('_', ' ')
  p = (gg.ggplot(df)
       + gg.aes(x=group_name, y='average_regret',
                colour='average_regret < {}'.format(regret_thresh))
       + gg.geom_point(size=5, alpha=0.8)
       + gg.scale_x_log10(breaks=[1, 3, 10, 30, 100])
       + gg.scale_colour_manual(values=['#d73027', '#313695'])
       + gg.ylab('average regret at {} episodes'.format(episode))
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
      )
  return facet_sweep_plot(p, sweep_vars)


def _make_unique_group_col(
    df: pd.DataFrame,
    sweep_vars: Optional[Sequence[str]] = None) -> pd.DataFrame:
  """Adds a unique_group column based on sweep_vars + bsuite_id."""
  unique_vars = ['bsuite_id']
  if sweep_vars:
    unique_vars += sweep_vars
  unique_group = (df[unique_vars].astype(str)
                  .apply(lambda x: x.name + '=' + x, axis=0)
                  .apply(lambda x: '\n'.join(x), axis=1)  # pylint:disable=unnecessary-lambda
                 )
  return unique_group


def plot_individual_returns(
    df_in: pd.DataFrame,
    max_episode: int,
    return_column: str = 'episode_return',
    colour_var: Optional[str] = None,
    yintercept: Optional[float] = None,
    sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plot individual learning curves: one curve per sweep setting."""
  df = df_in.copy()
  df['unique_group'] = _make_unique_group_col(df, sweep_vars)
  p = (gg.ggplot(df)
       + gg.aes(x='episode', y=return_column, group='unique_group')
       + gg.coord_cartesian(xlim=(0, max_episode))
      )
  if colour_var:
    p += gg.geom_line(gg.aes(colour=colour_var), size=1.1, alpha=0.75)
    if len(df[colour_var].unique()) <= 5:
      df[colour_var] = df[colour_var].astype('category')
      p += gg.scale_colour_manual(values=FIVE_COLOURS)
  else:
    p += gg.geom_line(size=1.1, alpha=0.75, colour='#313695')
  if yintercept:
    p += gg.geom_hline(
        yintercept=yintercept, alpha=0.5, size=2, linetype='dashed')
  return facet_sweep_plot(p, sweep_vars, tall_plot=True)
