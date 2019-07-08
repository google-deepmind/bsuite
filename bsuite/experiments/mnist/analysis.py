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
"""Analysis for MNIST."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.mnist import sweep
from bsuite.utils import plotting
import pandas as pd
import plotnine as gg
from typing import Text, Sequence

BASE_REGRET = 1.8
TAGS = ('basic', 'generalization')


def score(df: pd.DataFrame) -> float:
  """Output a single score for bandit experiment."""
  return plotting.ave_regret_score(
      df, baseline_regret=BASE_REGRET, episode=sweep.NUM_EPISODES)


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret through time."""
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=sweep.NUM_EPISODES)
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p
