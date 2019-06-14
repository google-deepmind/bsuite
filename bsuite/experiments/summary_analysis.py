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

from bsuite import plotting
from bsuite.experiments.bandit_noise import analysis as bandit_noise_analysis
from bsuite.experiments.bandit_scale import analysis as bandit_scale_analysis
from bsuite.experiments.catch import analysis as catch_analysis
from bsuite.experiments.deep_sea import analysis as deep_sea_analysis
from bsuite.experiments.discounting_chain import analysis as discounting_chain_analysis
from bsuite.experiments.memory_len import analysis as memory_chain_analysis
from bsuite.experiments.umbrella_distract import analysis as umbrella_distract_analysis
from bsuite.experiments.umbrella_length import analysis as umbrella_length_analysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Sequence, Text


def _bsuite_score_single(df: pd.DataFrame,
                         verbose: bool = False) -> pd.DataFrame:
  """Score the bsuite across all domains for a single agent."""
  data = []
  for env_name, env_data in df.groupby('bsuite_env'):
    if env_name not in BSUITE_ANALYSIS:
      if verbose:
        print('WARNING: {}_score not found in load.py and so is excluded.'
              .format(env_name))
    else:
      data.append({
          'bsuite_env': env_name,
          # BSUITE_ANALYSIS mapping defined at end of file.
          'score': BSUITE_ANALYSIS[env_name].score(env_data),
          'type': BSUITE_ANALYSIS[env_name].type,
          'finished': BSUITE_ANALYSIS[env_name].finished(env_data),
      })
  return pd.DataFrame(data)


def bsuite_score(df: pd.DataFrame, sweep_vars: Sequence[Text]) -> pd.DataFrame:
  """Score bsuite across hyperparameter settings."""
  if sweep_vars:
    return df.groupby(sweep_vars).apply(_bsuite_score_single).reset_index()
  else:
    return _bsuite_score_single(df)


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

  return plotting.facet_sweep_plot(p, sweep_vars)


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
       + gg.theme(axis_text_x=gg.element_text(angle=25, hjust=1))
       + gg.ylab('score')
       + gg.theme(axis_text_x=gg.element_blank(), figure_size=(21, 16))
      )
  if not all(df.finished):  # add a layer of alpha for unfinished jobs
    p += gg.aes(alpha='finished')
    p += gg.scale_alpha_discrete([0.3, 1.0])

  p += gg.facet_wrap(['env', 'type'], labeller='label_both')
  return p


def _radar(df, ax, label, all_types):
  """Plot utility."""
  tmp = df.groupby('type').mean().reset_index()

  values = []
  for curr_type in all_types:
    score = 0.
    selected = tmp[tmp['type'] == curr_type]
    if len(selected) == 1:
      score = float(selected['score'])
    else:
      print('{} bsuite scores found for type {!r} with setting {!r}. '
            'Replacing with zero.'.format(len(selected), curr_type, label))
    values.append(score)
  values = np.maximum(values, 0.05)  # don't let radar collapse to 0.
  values = np.concatenate((values, [values[0]]))

  angles = np.linspace(0, 2*np.pi, len(all_types), endpoint=False)
  angles = np.concatenate((angles, [angles[0]]))

  ax.plot(angles, values, 'o-', linewidth=2, label=label)
  ax.fill(angles, values, alpha=0.25)
  ax.set_thetagrids(angles * 180/np.pi, all_types)


def bsuite_radar_plot(
    score_data: pd.DataFrame, sweep_vars: Sequence[Text] = None):
  """Output a radar plot of bsuite data."""
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, polar=True)
  all_types = sorted(score_data['type'].unique())

  score_data['agent'] = (score_data[sweep_vars].astype(str)
                         .apply(lambda x: x.name + '=' + x, axis=0)
                         .apply(lambda x: '\n'.join(x), axis=1)  # pylint:disable=unnecessary-lambda
                        )

  if sweep_vars:
    for agent, sweep_df in score_data.groupby('agent'):
      _radar(sweep_df, ax, agent, all_types)
    ax.legend(bbox_to_anchor=(1.85, 1.2))
  else:
    _radar(score_data, ax, '', all_types)

  plt.yticks([0, 0.25, 0.5, 0.75, 1])
  ax.grid(True)


def _is_finished(df: pd.DataFrame, time_var: Text, n_min: float) -> bool:
  """Check to see if every wid in the dataframe is finished."""
  max_time = df.groupby('wid')[time_var].max().reset_index()
  return max_time[time_var].min() >= n_min


def basic_finished(df: pd.DataFrame) -> bool:
  return _is_finished(df, 'episode', 10000.)


def ou_finished(df: pd.DataFrame) -> bool:
  return _is_finished(df, 'steps', 10000.)


BsuiteScore = collections.namedtuple('BsuiteScore',
                                     ['score', 'type', 'finished'])

BSUITE_ANALYSIS = {
    'catch': BsuiteScore(
        catch_analysis.score, 'basic', basic_finished),
    'deep_sea': BsuiteScore(
        deep_sea_analysis.score, 'exploration', basic_finished),
    'bandit_scale': BsuiteScore(
        bandit_scale_analysis.score, 'scale', basic_finished),
    'bandit_noise': BsuiteScore(
        bandit_noise_analysis.score, 'noise', basic_finished),
    'memory_len': BsuiteScore(
        memory_chain_analysis.score, 'memory', basic_finished),
    'discounting_chain': BsuiteScore(
        discounting_chain_analysis.score, 'credit_assignment', basic_finished),
    'umbrella_length': BsuiteScore(
        umbrella_length_analysis.score, 'credit_assignment',
        basic_finished),
    'umbrella_distract': BsuiteScore(
        umbrella_distract_analysis.score, 'credit_assignment',
        basic_finished),
}
