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
"""Collection of smoothers designed for plotnine ggplot."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats


def _mean(data: pd.DataFrame, span: float, edge_tolerance: float = 0.):
  """Compute rolling mean of data via histogram, smooth endpoints.

  Args:
    data: pandas dataframe including columns ['x', 'y'] sorted by 'x'
    span: float in (0, 1) proportion of data to include.
    edge_tolerance: float of how much forgiveness to give to points that are
      close to the histogram boundary (in proportion of bin width).

  Returns:
    output_data: pandas dataframe with 'x', 'y' and 'stderr'
  """
  num_bins = np.ceil(1. / span).astype(np.int32)
  count, edges = np.histogram(data.x, bins=num_bins)

  # Include points that may be slightly on wrong side of histogram bin.
  tol = edge_tolerance * (edges[1] - edges[0])

  x_list = []
  y_list = []
  stderr_list = []
  for i, num_obs in enumerate(count):
    if num_obs > 0:
      sub_df = data.loc[(data.x > edges[i] - tol)
                        & (data.x < edges[i + 1] + tol)]
      x_list.append(sub_df.x.mean())
      y_list.append(sub_df.y.mean())
      stderr_list.append(sub_df.y.std() / np.sqrt(len(sub_df)))

  return pd.DataFrame(dict(x=x_list, y=y_list, stderr=stderr_list))


def mean(data: pd.DataFrame,
         xseq,
         span: float = 0.1,
         se: bool = True,
         level: float = 0.95,
         method_args: Optional[Dict[str, Any]] = None,
         **params) -> pd.DataFrame:
  """Computes the rolling mean over a portion of the data.

  Confidence intervals are given by approx Gaussian standard error bars.
  Unused/strangely named arguments are kept here for consistency with the rest
  of the plotnine package.

  Args:
    data: pandas dataframe passed to the smoother
    xseq: sequence of x at which to output prediction (unused)
    span: proportion of the data to use in lowess smoother.
    se: boolean for whether to show confidence interval.
    level: level in (0,1) for confidence standard errorbars
    method_args: other parameters that get passed through plotnine to method
      (edge_tolerance=0.05, num_boot=20)
    **params: dictionary other parameters passed to smoother (unused)

  Returns:
    output_data: pd Dataframe with x, y, ymin, ymax for confidence smooth.
  """
  del xseq  # Unused.
  del params  # Unused.

  if method_args is None:
    method_args = {}

  edge_tolerance = method_args.get('edge_tolerance', 0.05)
  output_data = _mean(data, span, edge_tolerance)

  if not se:
    return output_data

  num_std = stats.norm.interval(level)[1]  # Gaussian approx to CIs

  if 'group_smooth' in data.columns:
    # Perform bootstrapping over whole line/timeseries at once. Each unique
    # element of 'group_smooth' is treated as an atomic unit for bootstrap.
    data = data.set_index('group_smooth')
    num_boot = method_args.get('num_boot', 20)
    unique_ids = data.index.unique()
    boot_preds = np.ones([len(output_data), num_boot]) * np.nan

    for n in range(num_boot):
      boot_inds = np.random.choice(unique_ids, len(unique_ids))
      boot_data = data.loc[boot_inds].copy()
      boot_data = boot_data.sort_values('x')
      boot_out = _mean(boot_data, span, edge_tolerance)
      boot_preds[:, n] = np.interp(output_data.x, boot_out.x, boot_out.y)

    stddev = np.std(boot_preds, axis=1, ddof=2)
    output_data['ymin'] = output_data.y - num_std * stddev
    output_data['ymax'] = output_data.y + num_std * stddev

  else:
    # Just use the "estimated stderr" from each bin 1 / sqrt(n)
    output_data['ymin'] = output_data.y - num_std * output_data.stderr
    output_data['ymax'] = output_data.y + num_std * output_data.stderr

  return output_data

