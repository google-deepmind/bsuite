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
"""Functions for recording times spent in environments."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

import functools
import time
import pandas as pd
from typing import List, Tuple, Text


def time_run(func):
  """Meant for wrapping bsuite.baselines.scripts.run_all.run."""
  @functools.wraps(func)
  def wrapper(*args) -> Tuple[Text, float, Any]:
    bsuite_id = args[0][0]
    environment_name = bsuite_id.split('/')[0]
    t1 = time.time()
    func_result = func(*args)
    t2 = time.time()
    return environment_name, t2 - t1, func_result
  return wrapper


def extract_df(envname_and_duration: List[Tuple[Text, float]]) -> pd.DataFrame:
  df = pd.DataFrame(envname_and_duration, columns=['Environment', 'Duration'])
  df = df.groupby('Environment')['Duration'].agg(
      ['sum', 'mean', 'std', 'max', 'min', 'count'])
  df = df.sort_values(by=['sum'], ascending=False)
  return df
