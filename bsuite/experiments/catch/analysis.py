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

from bsuite import plotting
from bsuite.utils import smoothers

import numpy as np
import pandas as pd
import plotnine as gg

from typing import Sequence, Text


def score(df: pd.DataFrame) -> float:
  """Output a single score for catch."""
  n_eps = np.minimum(df.episode.max(), 10000)
  mean_regret = df.loc[df.episode == n_eps, 'total_regret'].mean() / n_eps
  return np.clip(1 - mean_regret, 0, 1)


def plot(df: pd.DataFrame,
         sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Simple learning curves for catch."""
  p = (gg.ggplot(df)
       + gg.aes('episode', 'mean_regret')
       + gg.geom_smooth(method=smoothers.mean, span=0.05, size=2, alpha=0.1,
                        colour='#313695', fill='#313695')
       + gg.geom_hline(
           gg.aes(yintercept=1.), linetype='dashed', alpha=0.4, size=1.75)
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
       + gg.ylab('average regret')
      )
  return plotting.facet_sweep_plot(p, sweep_vars)
