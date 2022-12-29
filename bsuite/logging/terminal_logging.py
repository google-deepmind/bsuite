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
"""A simple logger that pretty-prints to terminal."""

import logging as std_logging
import numbers
from typing import Any, Mapping

from absl import logging
from bsuite import environments
from bsuite.logging import base
from bsuite.utils import wrappers
import dm_env


def wrap_environment(env: environments.Environment,
                     pretty_print: bool = True,
                     log_every: bool = False,
                     log_by_step: bool = False) -> dm_env.Environment:
  """Returns a wrapped environment that logs to terminal."""
  # Set logging up to show up in STDERR.
  std_logging.getLogger().addHandler(logging.PythonHandler())
  logger = Logger(pretty_print, absl_logging=True)
  return wrappers.Logging(
      env, logger, log_by_step=log_by_step, log_every=log_every)


class Logger(base.Logger):
  """Writes data to terminal."""

  def __init__(self, pretty_print: bool = True, absl_logging: bool = False):
    self._pretty_print = pretty_print
    self._print_fn = logging.info if absl_logging else print

  def write(self, data: Mapping[str, Any]):
    """Writes to terminal, pretty-printing the results."""

    if self._pretty_print:
      data = pretty_dict(data)

    self._print_fn(data)


def pretty_dict(data: Mapping[str, Any]) -> str:
  """Prettifies a dictionary into a string as `k1 = v1 | ... | kn = vn`."""
  msg = []
  for key in sorted(data):
    value = value_format(data[key])
    msg_pair = f'{key} = {value}'
    msg.append(msg_pair)

  return ' | '.join(msg)


def value_format(value: Any) -> str:
  """Convenience function for string formatting."""
  if isinstance(value, numbers.Integral):
    return str(value)
  if isinstance(value, numbers.Number):
    return f'{value:0.4f}'
  return str(value)
