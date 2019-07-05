# pylint: disable=g-bad-file-header
# Copyright 2019 The bsuite Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

import collections

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
from bsuite.experiments.hierarchy_sea import analysis as hierarchy_sea_analysis
from bsuite.experiments.hierarchy_sea_explore import analysis as hierarchy_sea_explore_analysis
from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import analysis as memory_size_analysis
from bsuite.experiments.mnist import analysis as mnist_analysis
from bsuite.experiments.mnist_noise import analysis as mnist_noise_analysis
from bsuite.experiments.mnist_scale import analysis as mnist_scale_analysis
from bsuite.experiments.mountain_car import analysis as mountain_car_analysis
from bsuite.experiments.mountain_car_noise import analysis as mountain_car_noise_analysis
from bsuite.experiments.mountain_car_scale import analysis as mountain_car_scale_analysis
from bsuite.experiments.nonstationary_bandit import analysis as nonstationary_bandit_analysis
from bsuite.experiments.nonstationary_rps import analysis as nonstationary_rps_analysis
from bsuite.experiments.umbrella_distract import analysis as umbrella_distract_analysis
from bsuite.experiments.umbrella_length import analysis as umbrella_length_analysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Mapping, Sequence, Text


################################################################################
# Summarizing scores
BsuiteSummary = collections.namedtuple(
    'BsuiteSummary', ['score', 'type', 'tags', 'episode'])


def _parse_bsuite(package) -> BsuiteSummary:
  """Returns a Bsuite summary from a package."""
  return BsuiteSummary(
      score=package.score,
      type=package.TAGS[0],
      tags=package.TAGS,
      episode=package.EPISODE,
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
    hierarchy_sea=_parse_bsuite(hierarchy_sea_analysis),
    hierarchy_sea_explore=_parse_bsuite(hierarchy_sea_explore_analysis),
    memory_len=_parse_bsuite(memory_len_analysis),
    memory_size=_parse_bsuite(memory_size_analysis),
    mnist=_parse_bsuite(mnist_analysis),
    mnist_noise=_parse_bsuite(mnist_noise_analysis),
    mnist_scale=_parse_bsuite(mnist_scale_analysis),
    mountain_car=_parse_bsuite(mountain_car_analysis),
    mountain_car_noise=_parse_bsuite(mountain_car_noise_analysis),
    mountain_car_scale=_parse_bsuite(mountain_car_scale_analysis),
    nonstationary_bandit=_parse_bsuite(nonstationary_bandit_analysis),
    nonstationary_rps=_parse_bsuite(nonstationary_rps_analysis),
    umbrella_distract=_parse_bsuite(umbrella_distract_analysis),
    umbrella_length=_parse_bsuite(umbrella_length_analysis),
)

ALL_TAGS = set()
for bsuite_summary in BSUITE_INFO.values():
  ALL_TAGS = ALL_TAGS.union(set(bsuite_summary.tags))


def _is_finished(df: pd.DataFrame, n_min: int) -> bool:
  """Check to see if every bsuite id in the dataframe is finished."""
  # At this point we have grouped by any additional hyperparameters.
  if 'wid' in df.columns:
    assert len(set(df.bsuite_id)) == len(set(df.wid))
  max_time = df.groupby('bsuite_id')['episode'].max().reset_index()
  return max_time['episode'].min() >= n_min


def _bsuite_score_single(df: pd.DataFrame,
                         experiment_info: Mapping[Text, BsuiteSummary],
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
                 sweep_vars: Sequence[Text]) -> pd.DataFrame:
  """Score bsuite for each experiment across hyperparameter settings."""
  score_fun = lambda x: _bsuite_score_single(x, BSUITE_INFO)
  if sweep_vars:
    score_df = df.groupby(sweep_vars).apply(score_fun).reset_index()
  else:
    score_df = score_fun(df)
  return score_df


def _summarize_single_by_tag(score_df: pd.DataFrame,
                             unique_tags: Sequence[Text],
                             tags_column: Text) -> pd.DataFrame:
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
                     sweep_vars: Sequence[Text]) -> pd.DataFrame:
  """Takes in a bsuite scored dataframe and summarizes by tags."""
  summary_fun = lambda x: _summarize_single_by_tag(x, list(ALL_TAGS), 'tags')
  if sweep_vars:
    summary_df = score_df.groupby(sweep_vars).apply(summary_fun).reset_index()
  else:
    summary_df = summary_fun(score_df)
  return  summary_df


################################################################################
# Summary plots


def bsuite_bar_plot(df_in: pd.DataFrame,
                    sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Output bar plot of bsuite data."""
  df = df_in.copy()
  env_categories = df.sort_values('type').bsuite_env.drop_duplicates().values
  df['clean_env'] = pd.Categorical(
      df.bsuite_env, categories=env_categories, ordered=True)

  p = (gg.ggplot(df)
       + gg.aes(x='clean_env', y='score', colour='type', fill='type')
       + gg.geom_bar(position='dodge', stat='identity')
       + gg.ylim(0, 1)
       + gg.xlab('challenge')
       + gg.theme(axis_text_x=gg.element_text(angle=25, hjust=1))
       + gg.ylab('score')
      )
  if not all(df.finished):  # add a layer of alpha for unfinished jobs
    p += gg.aes(alpha='finished')
    p += gg.scale_alpha_discrete([0.3, 1.0])

  # Compute the necessary size of the plot
  if sweep_vars:
    p += gg.facet_wrap(sweep_vars, labeller='label_both', ncol=1)
    n_hypers = df[sweep_vars].drop_duplicates().shape[0]
  else:
    n_hypers = 1
  return p + gg.theme(figure_size=(16, 4 * n_hypers + 1))


def bsuite_bar_plot_compare(df_in: pd.DataFrame,
                            sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Output bar plot of bsuite data."""
  df = df_in.copy()
  env_categories = df.sort_values('type').bsuite_env.drop_duplicates().values
  df['env'] = pd.Categorical(
      df.bsuite_env, categories=env_categories, ordered=True)

  df['agent'] = (df[sweep_vars].astype(str)
                 .apply(lambda x: x.name + '=' + x, axis=0)
                 .apply(lambda x: '\n'.join(x), axis=1)  # pylint:disable=unnecessary-lambda
                )

  p = (gg.ggplot(df)
       + gg.aes(x='agent', y='score', colour='agent', fill='agent')
       + gg.geom_bar(position='dodge', stat='identity')
       + gg.ylim(0, 1)
       + gg.xlab('agent')
       + gg.theme(axis_text_x=gg.element_text(angle=45, hjust=1))
       + gg.ylab('score')
       + gg.theme(axis_text_x=gg.element_blank(), figure_size=(21, 16))
      )
  if not all(df.finished):  # add a layer of alpha for unfinished jobs
    p += gg.aes(alpha='finished')
    p += gg.scale_alpha_discrete([0.3, 1.0])

  p += gg.facet_wrap(['env', 'type'], labeller='label_both')
  return p


def _radar(df, ax, label, all_tags):
  """Plot utility."""
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

  ax.plot(angles, values, 'o-', linewidth=2, label=label)
  ax.fill(angles, values, alpha=0.25)
  ax.set_thetagrids(angles * 180/np.pi, all_tags)


def bsuite_radar_plot(summary_data: pd.DataFrame,
                      sweep_vars: Sequence[Text] = None):
  """Output a radar plot of bsuite data from bsuite_summary by tag."""
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, polar=True)
  all_tags = sorted(summary_data['tag'].unique())

  summary_data['agent'] = (summary_data[sweep_vars].astype(str)
                           .apply(lambda x: x.name + '=' + x, axis=0)
                           .apply(lambda x: '\n'.join(x), axis=1)  # pylint:disable=unnecessary-lambda
                          )

  if sweep_vars:
    for agent, sweep_df in summary_data.groupby('agent'):
      _radar(sweep_df, ax, agent, all_tags)
    ax.legend(bbox_to_anchor=(1.85, 1.2))
  else:
    _radar(summary_data, ax, '', all_tags)

  plt.yticks([0, 0.25, 0.5, 0.75, 1])
  ax.grid(True)
