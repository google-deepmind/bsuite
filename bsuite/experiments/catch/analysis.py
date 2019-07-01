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
"""Analysis for catch."""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.utils import plotting
import pandas as pd
import plotnine as gg

from typing import Sequence, Text


BASE_REGRET = 1.
EPISODE = 10000
TAGS = ('basic',)


def score(df: pd.DataFrame) -> float:
  """Output a single score for catch."""
  return plotting.ave_regret_score(
      df, baseline_regret=BASE_REGRET, episode=EPISODE)


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Simple learning curves for catch."""
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=EPISODE)
  p += gg.geom_hline(
      gg.aes(yintercept=BASE_REGRET), linetype='dashed', alpha=0.4, size=1.75)
  return p
