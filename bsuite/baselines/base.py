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
"""A simple agent interface."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

import abc
import dm_env

# pylint: disable=invalid-name
Action = int  # Only discrete-action agents for now.
# pylint: enable=invalid-name


class Agent(object):
  """An agent consists of a policy and an update rule."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def policy(self, timestep: dm_env.TimeStep) -> Action:
    """A policy takes in a timestep and returns an action."""

  @abc.abstractmethod
  def update(
      self,
      timestep: dm_env.TimeStep,
      action: Action,
      new_timestep: dm_env.TimeStep,
  ) -> None:
    """Updates the agent given a transition."""
