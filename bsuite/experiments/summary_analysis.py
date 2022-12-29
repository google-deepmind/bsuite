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
"""Plots for summary data across all experiments, e.g. the radar plot."""

from typing import Callable, Mapping, NamedTuple, Optional, Sequence, Union

from bsuite.experiments.bandit import analysis as bandit_analysis
from bsuite.experiments.bandit_noise import analysis as bandit_noise_analysis
from bsuite.experiments.bandit_scale import analysis as bandit_scale_analysis
from bsuite.experiments.cartpole import analysis as cartpole_analysis
from bsuite.experiments.cartpole_noise import analysis as cartpole_noise_analysis
from bsuite.experiments.cartpole_scale import analysis as cartpole_scale_analysis
from bsuite.experiments.cartpole_swingup import analysis as cartpole_swingup_analysis
from bsuite.experiments.catch import analysis as catch_analysis
from bsuite.experiments.catch_noise import analysis as catch_noise_analysis
from bsuite.experiments.catch_scale import analysis as catch_scale_analysis
from bsuite.experiments.deep_sea import analysis as deep_sea_analysis
from bsuite.experiments.deep_sea_stochastic import analysis as deep_sea_stochastic_analysis
from bsuite.experiments.discounting_chain import analysis as discounting_chain_analysis
from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import analysis as memory_size_analysis
from bsuite.experiments.mnist import analysis as mnist_analysis
from bsuite.experiments.mnist_noise import analysis as mnist_noise_analysis
from bsuite.experiments.mnist_scale import analysis as mnist_scale_analysis
from bsuite.experiments.mountain_car import analysis as mountain_car_analysis
from bsuite.experiments.mountain_car_noise import analysis as mountain_car_noise_analysis
from bsuite.experiments.mountain_car_scale import analysis as mountain_car_scale_analysis
from bsuite.experiments.umbrella_distract import analysis as umbrella_distract_analysis
from bsuite.experiments.umbrella_length import analysis as umbrella_length_analysis
from bsuite.utils import plotting

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg


################################################################################
# Summarizing scores


class BSuiteSummary(NamedTuple):
  """Container for summary metadata for a given bsuite experiment."""
  score: Callable[[pd.DataFrame], float]
  type: str
  tags: Sequence[str]
  episode: int


def _parse_bsuite(package) -> BSuiteSummary:
  """Returns a Bsuite summary from a package."""
  return BSuiteSummary(
      score=package.score,
      type=package.TAGS[0],
      tags=package.TAGS,
      episode=package.NUM_EPISODES,
      )

BSUITE_INFO = dict(
    bandit=_parse_bsuite(bandit_analysis),
    bandit_noise=_parse_bsuite(bandit_noise_analysis),
    bandit_scale=_parse_bsuite(bandit_scale_analysis),
    cartpole=_parse_bsuite(cartpole_analysis),
    cartpole_noise=_parse_bsuite(cartpole_noise_analysis),
    cartpole_scale=_parse_bsuite(cartpole_scale_analysis),
    cartpole_swingup=_parse_bsuite(cartpole_swingup_analysis),
    catch=_parse_bsuite(catch_analysis),
    catch_noise=_parse_bsuite(catch_noise_analysis),
    catch_scale=_parse_bsuite(catch_scale_analysis),
    deep_sea=_parse_bsuite(deep_sea_analysis),
    deep_sea_stochastic=_parse_bsuite(deep_sea_stochastic_analysis),
    discounting_chain=_parse_bsuite(discounting_chain_analysis),
    memory_len=_parse_bsuite(memory_len_analysis),
    memory_size=_parse_bsuite(memory_size_analysis),
    mnist=_parse_bsuite(mnist_analysis),
    mnist_noise=_parse_bsuite(mnist_noise_analysis),
    mnist_scale=_parse_bsuite(mnist_scale_analysis),
    mountain_car=_parse_bsuite(mountain_car_analysis),
    mountain_car_noise=_parse_bsuite(mountain_car_noise_analysis),
    mountain_car_scale=_parse_bsuite(mountain_car_scale_analysis),
    umbrella_distract=_parse_bsuite(umbrella_distract_analysis),
    umbrella_length=_parse_bsuite(umbrella_length_analysis),
)

ALL_TAGS = set()
for bsuite_summary in BSUITE_INFO.values():
  ALL_TAGS = ALL_TAGS.union(set(bsuite_summary.tags))


def _is_finished(df: pd.DataFrame, n_min: int) -> bool:
  """Check to see if every bsuite id in the dataframe is finished."""
  # At this point we have grouped by any additional hyperparameters.
  # Check if we have run enough episodes for every id.
  max_time = df.groupby('bsuite_id')['episode'].max().reset_index()
  return max_time['episode'].min() >= n_min


def _bsuite_score_single(df: pd.DataFrame,
                         experiment_info: Mapping[str, BSuiteSummary],
                         verbose: bool = False) -> pd.DataFrame:
  """Score the bsuite across all domains for a single agent."""
  data = []
  for env_name, env_data in df.groupby('bsuite_env'):
    if env_name not in experiment_info:
      if verbose:
        print('WARNING: {}_score not found in load.py and so is excluded.'
              .format(env_name))
    else:
      b_summary = experiment_info[env_name]
      data.append({
          'bsuite_env': env_name,
          'score': b_summary.score(env_data),
          'type': b_summary.type,
          'tags': str(b_summary.tags),
          'finished': _is_finished(env_data, b_summary.episode),
      })
  return pd.DataFrame(data)


def bsuite_score(df: pd.DataFrame,
                 sweep_vars: Optional[Sequence[str]] = None) -> pd.DataFrame:
  """Score bsuite for each experiment across hyperparameter settings."""
  score_fun = lambda x: _bsuite_score_single(x, BSUITE_INFO)
  if sweep_vars:
    score_df = df.groupby(sweep_vars).apply(score_fun).reset_index()
  else:
    score_df = score_fun(df)

  # Groupby has a habit of adding meaningless columns to dataframe.
  for col in df.columns:
    if col in ['level_0', 'level_1', 'level_2']:
      score_df.drop(col, axis=1, inplace=True)
  return score_df


def _summarize_single_by_tag(score_df: pd.DataFrame,
                             unique_tags: Sequence[str],
                             tags_column: str) -> pd.DataFrame:
  """Takes in a single scored dataframe and averages score over tags."""
  df = score_df.copy()
  # Expand the columns of dataframe to indicate if it contains valid tag.
  for tag in unique_tags:
    df[tag] = df[tags_column].str.contains(tag)

  data = []
  for tag in unique_tags:
    ave_score = df.loc[df[tag], 'score'].mean()
    data.append({'tag': tag, 'score': ave_score})
  return pd.DataFrame(data)


def ave_score_by_tag(score_df: pd.DataFrame,
                     sweep_vars: Sequence[str]) -> pd.DataFrame:
  """Takes in a bsuite scored dataframe and summarizes by tags."""
  summary_fun = lambda x: _summarize_single_by_tag(x, list(ALL_TAGS), 'tags')
  if sweep_vars:
    summary_df = score_df.groupby(sweep_vars).apply(summary_fun).reset_index()
  else:
    summary_df = summary_fun(score_df)
  return  summary_df


################################################################################
# Summary plots


def _gen_ordered_experiments() -> Sequence[str]:
  """Provides a list of ordered experiments for bar plot."""
  basics = ['bandit', 'mnist', 'catch', 'mountain_car', 'cartpole']
  noise = [env + '_noise' for env in basics]
  scale = [env + '_scale' for env in basics]
  explore = ['deep_sea', 'deep_sea_stochastic', 'cartpole_swingup']
  credit = ['umbrella_length', 'umbrella_distract', 'discounting_chain']
  memory = ['memory_len', 'memory_size']
  return basics + noise + scale + explore + credit + memory

_ORDERED_EXPERIMENTS = _gen_ordered_experiments()
_ORDERED_TYPES = [
    'basic', 'noise', 'scale', 'exploration', 'credit_assignment', 'memory']


def _clean_bar_plot_data(
    df_in: pd.DataFrame,
    sweep_vars: Optional[Sequence[str]] = None) -> pd.DataFrame:
  """Clean the summary data for bar plot comparison of agents."""
  df = df_in.copy()
  df['env'] = pd.Categorical(
      df.bsuite_env, categories=_ORDERED_EXPERIMENTS, ordered=True)
  df['type'] = pd.Categorical(
      df['type'], categories=_ORDERED_TYPES, ordered=True)

  if sweep_vars is None:
    df['agent'] = 'agent'
  elif len(sweep_vars) == 1:
    df['agent'] = df[sweep_vars[0]].astype(str)
  else:
    df['agent'] = (df[sweep_vars].astype(str)
                   .apply(lambda x: x.name + '=' + x, axis=0)
                   .apply(lambda x: '\n'.join(x), axis=1)  # pylint:disable=unnecessary-lambda
                  )
  return df


def bsuite_bar_plot(df_in: pd.DataFrame,
                    sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Output bar plot of bsuite data."""
  df = _clean_bar_plot_data(df_in, sweep_vars)

  p = (gg.ggplot(df)
       + gg.aes(x='env', y='score', colour='type', fill='type')
       + gg.geom_bar(position='dodge', stat='identity')
       + gg.geom_hline(yintercept=1., linetype='dashed', alpha=0.5)
       + gg.scale_colour_manual(plotting.CATEGORICAL_COLOURS)
       + gg.scale_fill_manual(plotting.CATEGORICAL_COLOURS)
       + gg.xlab('experiment')
       + gg.theme(axis_text_x=gg.element_text(angle=25, hjust=1))
      )
  if not all(df.finished):  # add a layer of alpha for unfinished jobs
    p += gg.aes(alpha='finished')
    p += gg.scale_alpha_discrete(range=[0.3, 1.0])

  # Compute the necessary size of the plot
  if sweep_vars:
    p += gg.facet_wrap(sweep_vars, labeller='label_both', ncol=1)
    n_hypers = df[sweep_vars].drop_duplicates().shape[0]
  else:
    n_hypers = 1
  return p + gg.theme(figure_size=(14, 3 * n_hypers + 1))


def _bar_plot_compare(df: pd.DataFrame) -> gg.ggplot:
  """Bar plot of buite score data, comparing agents on each experiment."""
  p = (gg.ggplot(df)
       + gg.aes(x='agent', y='score', colour='agent', fill='agent')
       + gg.geom_bar(position='dodge', stat='identity')
       + gg.geom_hline(yintercept=1., linetype='dashed', alpha=0.5)
       + gg.theme(axis_text_x=gg.element_text(angle=25, hjust=1))
       + gg.scale_colour_manual(plotting.CATEGORICAL_COLOURS)
       + gg.scale_fill_manual(plotting.CATEGORICAL_COLOURS)
      )
  if not all(df.finished):  # add a layer of alpha for unfinished jobs
    p += gg.aes(alpha='finished')
    p += gg.scale_alpha_discrete(range=[0.3, 1.0])
  return p


def bsuite_bar_plot_compare(
    df_in: pd.DataFrame,
    sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Output bar plot of bsuite data, comparing agents on each experiment."""
  df = _clean_bar_plot_data(df_in, sweep_vars)
  p = _bar_plot_compare(df)
  p += gg.facet_wrap('env', labeller='label_both')
  p += gg.theme(figure_size=(18, 16))
  return p


def plot_single_experiment(
    summary_df: pd.DataFrame,
    bsuite_env: str,
    sweep_vars: Optional[Sequence[str]] = None) -> Union[gg.ggplot, None]:
  """Compare score for just one experiment."""
  if len(summary_df) == 0:  # pylint:disable=g-explicit-length-test
    print('WARNING: you have no bsuite summary data, please reload.')
    return
  env_df = summary_df[summary_df.bsuite_env == bsuite_env]
  if len(env_df) == 0:  # pylint:disable=g-explicit-length-test
    print('Warning, you have no data for bsuite_env={}'.format(bsuite_env))
    print('Your dataframe only includes bsuite_env={}'
          .format(summary_df.bsuite_env.unique()))
    return

  df = _clean_bar_plot_data(env_df, sweep_vars)
  n_agent = len(df.agent.unique())
  p = _bar_plot_compare(df)
  plot_width = min(2 + n_agent, 12)
  p += gg.theme(figure_size=(plot_width, 6))
  p += gg.ggtitle('bsuite score for {} experiment'.format(bsuite_env))
  print('tags={}'.format(df.tags.iloc[0]))
  return p


def _tag_pretify(tag):
  return tag.replace('_', ' ').title()


def _radar(
    df: pd.DataFrame, ax: plt.Axes, label: str, all_tags: Sequence[str],
    color: str, alpha: float = 0.2, edge_alpha: float = 0.85, zorder: int = 2,
    edge_style: str = '-'):
  """Plot utility for generating the underlying radar plot."""
  tmp = df.groupby('tag').mean().reset_index()

  values = []
  for curr_tag in all_tags:
    score = 0.
    selected = tmp[tmp['tag'] == curr_tag]
    if len(selected) == 1:
      score = float(selected['score'])
    else:
      print('{} bsuite scores found for tag {!r} with setting {!r}. '
            'Replacing with zero.'.format(len(selected), curr_tag, label))
    values.append(score)
  values = np.maximum(values, 0.05)  # don't let radar collapse to 0.
  values = np.concatenate((values, [values[0]]))

  angles = np.linspace(0, 2*np.pi, len(all_tags), endpoint=False)
  angles = np.concatenate((angles, [angles[0]]))

  ax.plot(angles, values, '-', linewidth=5, label=label,
          c=color, alpha=edge_alpha, zorder=zorder, linestyle=edge_style)
  ax.fill(angles, values, alpha=alpha, color=color, zorder=zorder)
  # TODO(iosband): Necessary for some change in matplotlib code...
  axis_angles = angles[:-1] * 180/np.pi
  ax.set_thetagrids(
      axis_angles, map(_tag_pretify, all_tags), fontsize=18)

  # To avoid text on top of gridlines, we flip horizontalalignment
  # based on label location
  text_angles = np.rad2deg(angles)
  for label, angle in zip(ax.get_xticklabels()[:-1], text_angles[:-1]):
    if 90 <= angle <= 270:
      label.set_horizontalalignment('right')
    else:
      label.set_horizontalalignment('left')


def bsuite_radar_plot(summary_data: pd.DataFrame,
                      sweep_vars: Optional[Sequence[str]] = None):
  """Output a radar plot of bsuite data from bsuite_summary by tag."""
  fig = plt.figure(figsize=(8, 8), facecolor='white')

  ax = fig.add_subplot(111, polar=True)
  try:
    ax.set_axis_bgcolor('white')
  except AttributeError:
    ax.set_facecolor('white')
  all_tags = sorted(summary_data['tag'].unique())

  if sweep_vars is None:
    summary_data['agent'] = 'agent'
  elif len(sweep_vars) == 1:
    summary_data['agent'] = summary_data[sweep_vars[0]].astype(str)
  else:
    summary_data['agent'] = (summary_data[sweep_vars].astype(str)
                             .apply(lambda x: x.name + '=' + x, axis=0)
                             .apply(lambda x: '\n'.join(x), axis=1)  # pylint:disable=unnecessary-lambda
                            )
  if len(summary_data.agent.unique()) > 5:
    print('WARNING: We do not recommend radar plot for more than 5 agents.')

  # Creating radar plot background by hand, reusing the _radar call
  # it will give a slight illusion of being "3D" as inner part will be
  # darker than the outer
  thetas = np.linspace(0, 2*np.pi, 100)
  ax.fill(thetas, [0.25,] * 100, color='k', alpha=0.05)
  ax.fill(thetas, [0.5,] * 100, color='k', alpha=0.05)
  ax.fill(thetas, [0.75,] * 100, color='k', alpha=0.03)
  ax.fill(thetas, [1.,] * 100, color='k', alpha=0.01)

  palette = lambda x: plotting.CATEGORICAL_COLOURS[x]
  if sweep_vars:
    sweep_data_ = summary_data.groupby('agent')
    for aid, (agent, sweep_df) in enumerate(sweep_data_):
      _radar(sweep_df, ax, agent, all_tags, color=palette(aid))
    if len(sweep_vars) == 1:
      label = sweep_vars[0]
      if label == 'experiment':
        label = 'agent'  # rename if actually each individual agent
      legend = ax.legend(loc=(1.1, 0.), ncol=1, title=label)
      ax.get_legend().get_title().set_fontsize('20')
      ax.get_legend().get_title().set_fontname('serif')
      ax.get_legend().get_title().set_color('k')
      ax.get_legend().get_title().set_alpha(0.75)
      legend._legend_box.align = 'left'  # pylint:disable=protected-access
    else:
      legend = ax.legend(loc=(1.1, 0.), ncol=1,)
    plt.setp(legend.texts, fontname='serif')
    frame = legend.get_frame()
    frame.set_color('white')
    for text in legend.get_texts():
      text.set_color('grey')
  else:
    _radar(summary_data, ax, '', all_tags, color=palette(0))

  # Changing internal lines to be dotted and semi transparent
  for line in ax.xaxis.get_gridlines():
    line.set_color('grey')
    line.set_alpha(0.95)
    line.set_linestyle(':')
    line.set_linewidth(2)

  for line in ax.yaxis.get_gridlines():
    line.set_color('grey')
    line.set_alpha(0.95)
    line.set_linestyle(':')
    line.set_linewidth(2)

  plt.xticks(color='grey', fontname='serif')
  ax.set_rlabel_position(0)
  plt.yticks(
      [0, 0.25, 0.5, 0.75, 1],
      ['', '.25', '.5', '.75', '1'],
      color='k', alpha=0.75, fontsize=16, fontname='serif')
  # For some reason axis labels are behind plot by default ...
  ax.set_axisbelow(False)
  return fig
