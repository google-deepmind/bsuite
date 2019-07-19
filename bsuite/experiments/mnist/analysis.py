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
"""Analysis for MNIST."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.mnist import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg
from typing import Text, Sequence

NUM_EPISODES = sweep.NUM_EPISODES
BASE_REGRET = 1.8
TAGS = ('basic', 'generalization')


def score(df: pd.DataFrame) -> float:
  """Output a single score = 50% regret, 50% "final accuracy"."""
  regret_score = plotting.ave_regret_score(
      df, baseline_regret=BASE_REGRET, episode=sweep.NUM_EPISODES)

  final_df = df.copy()
  final_df['ave_return'] = (
      final_df.total_return.diff() / final_df.episode.diff())
  final_df = final_df[final_df.episode > 0.9 * NUM_EPISODES]
  # Convert (+1, -1) average return --> (+1, 0) accuracy score
  acc_score = np.mean(final_df.ave_return + 1) * 0.5
  return 0.5 * (regret_score + acc_score)


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret through time."""
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=sweep.NUM_EPISODES)
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p
