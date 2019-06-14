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
# Import all packages

import numbers

from absl import flags

from bsuite.logging import base
from bsuite.utils import wrappers

import dm_env

from typing import Any, Mapping, Text

FLAGS = flags.FLAGS


def wrap_environment(env: dm_env.Environment,
                     pretty_print: bool = True,
                     log_every: bool = False,
                     log_by_step: bool = False) -> dm_env.Environment:
  """Returns a wrapped environment that logs to terminal."""
  logger = Logger(pretty_print)
  return wrappers.Logging(
      env, logger, log_by_step=log_by_step, log_every=log_every)


class Logger(base.Logger):
  """Writes data to terminal."""

  def __init__(self, pretty_print: bool = True):
    self._pretty_print = pretty_print

  def write(self, data: Mapping[Text, Any]):
    """Writes to terminal, pretty-printing the results."""

    if self._pretty_print:
      msg = []
      for key in sorted(data):
        value = data[key]
        msg_pair = '{} = {}'.format(key, value_format(value))
        msg.append(msg_pair)

      data = '{}\n'.format(' | '.join(msg))

    print(data)


def value_format(value: Any) -> Text:
  """Convenience function for string formatting."""
  if isinstance(value, numbers.Real):
    return '{:0.4f}'.format(value)
  return '{}'.format(value)
